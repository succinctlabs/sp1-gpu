use std::fs::File;
use std::io::{Seek, Write};
use std::sync::Arc;
use std::time::Instant;

use size::Size;

use sp1_core::air::{PublicValues, Word};
use sp1_core::utils::{SP1CoreProverError, SP1ProverOpts, DIGEST_SIZE};
use tracing::instrument;

use p3_baby_bear::BabyBear;

use p3_field::AbstractField;

use p3_challenger::CanObserve;

use sp1_core::stark::MachineRecord;
use sp1_core::stark::{Challenge, Challenger, MachineProof, ShardProof, Val};
use sp1_prover::{
    ReduceProgramType, SP1CoreProof, SP1CoreProofData, SP1DeferredMemoryLayout, SP1Prover,
    SP1ProvingKey, SP1PublicValues, SP1RecursionMemoryLayout, SP1ReduceProof, SP1Stdin,
    SP1VerifyingKey,
};
use sp1_recursion_program::hints::Hintable;
use sp1_recursion_program::machine::{
    SP1CompressVerifier, SP1DeferredVerifier, SP1RecursiveVerifier, SP1RootVerifier,
};

use sp1_core::runtime::{ExecutionError, ExecutionRecord, NoOpSubproofVerifier, SP1Context};
use sp1_core::runtime::{ExecutionReport, ShardingConfig};
use sp1_core::{
    runtime::{Program, Runtime},
    stark::{RiscvAir, StarkGenericConfig, StarkProvingKey, StarkVerifyingKey},
    utils::{BabyBearPoseidon2, SP1CoreOpts},
};

use sp1_recursion_core::{
    runtime::{RecursionProgram, Runtime as RecursionRuntime},
    stark::{config::BabyBearPoseidon2Outer, RecursionAir},
};

use sp1_recursion_compiler::config::InnerConfig;

use sp1_recursion_program::machine::SP1ReduceMemoryLayout;

use sp1_prover::SP1RecursionProverError;

use moongate_core::stark::StarkGpuProver;

/// The configuration for the core prover.
pub type CoreSC = BabyBearPoseidon2;

/// The configuration for the inner prover.
pub type InnerSC = BabyBearPoseidon2;

/// The configuration for the outer prover.
pub type OuterSC = BabyBearPoseidon2Outer;

const REDUCE_DEGREE: usize = 3;
const COMPRESS_DEGREE: usize = 9;
const WRAP_DEGREE: usize = 9;

pub type ReduceAir<F> = RecursionAir<F, REDUCE_DEGREE>;
pub type CompressAir<F> = RecursionAir<F, COMPRESS_DEGREE>;
pub type WrapAir<F> = RecursionAir<F, WRAP_DEGREE>;

pub struct SP1GpuProver {
    /// The program that can recursively verify a set of proofs into a single proof.
    pub recursion_program: RecursionProgram<BabyBear>,

    /// The proving key for the recursion step.
    pub rec_pk: StarkProvingKey<InnerSC>,

    /// The verification key for the recursion step.
    pub rec_vk: StarkVerifyingKey<InnerSC>,

    /// The program that recursively verifies deferred proofs and accumulates the digests.
    pub deferred_program: RecursionProgram<BabyBear>,

    /// The proving key for the reduce step.
    pub deferred_pk: StarkProvingKey<InnerSC>,

    /// The verification key for the reduce step.
    pub deferred_vk: StarkVerifyingKey<InnerSC>,

    /// The program that reduces a set of recursive proofs into a single proof.
    pub compress_program: RecursionProgram<BabyBear>,

    /// The proving key for the reduce step.
    pub compress_pk: StarkProvingKey<InnerSC>,

    /// The verification key for the reduce step.
    pub compress_vk: StarkVerifyingKey<InnerSC>,

    /// The shrink program that compresses a proof into a succinct proof.
    pub shrink_program: RecursionProgram<BabyBear>,

    /// The proving key for the compress step.
    pub shrink_pk: StarkProvingKey<InnerSC>,

    /// The verification key for the compress step.
    pub shrink_vk: StarkVerifyingKey<InnerSC>,

    /// The prover for the core machine.
    pub(crate) core_prover: StarkGpuProver<CoreSC, RiscvAir<<CoreSC as StarkGenericConfig>::Val>>,

    /// The prover for the compress machine.
    pub(crate) compress_prover:
        StarkGpuProver<InnerSC, ReduceAir<<InnerSC as StarkGenericConfig>::Val>>,

    /// The prover for the shrink machine.
    pub(crate) shrink_prover:
        StarkGpuProver<InnerSC, CompressAir<<InnerSC as StarkGenericConfig>::Val>>,
}

impl SP1GpuProver {
    /// Initializes a new [SP1GpuProver].
    #[instrument(name = "initialize prover", level = "debug", skip_all)]
    pub fn new() -> Self {
        let core_machine = RiscvAir::machine(CoreSC::default());
        let core_prover = StarkGpuProver::new(core_machine);

        // Get the recursive verifier and setup the proving and verifying keys.
        let recursion_program =
            SP1RecursiveVerifier::<InnerConfig, _>::build(core_prover.machine());
        let compress_machine = ReduceAir::machine(InnerSC::default());
        let (rec_pk, rec_vk) = compress_machine.setup(&recursion_program);
        let compress_prover = StarkGpuProver::new(compress_machine);

        // Get the deferred program and keys.
        let deferred_program =
            SP1DeferredVerifier::<InnerConfig, _, _>::build(compress_prover.machine());
        let (deferred_pk, deferred_vk) = compress_prover.setup(&deferred_program);

        // Make the reduce program and keys.
        let compress_program = SP1CompressVerifier::<InnerConfig, _, _>::build(
            compress_prover.machine(),
            &rec_vk,
            &deferred_vk,
        );
        let (compress_pk, compress_vk) = compress_prover.setup(&compress_program);

        // Get the compress program, machine, and keys.
        let shrink_program = SP1RootVerifier::<InnerConfig, _, _>::build(
            compress_prover.machine(),
            &compress_vk,
            RecursionProgramType::Shrink,
        );
        let shrink_machine = CompressAir::wrap_machine_dyn(InnerSC::compressed());
        let (shrink_pk, shrink_vk) = shrink_machine.setup(&shrink_program);
        let shrink_prover = StarkGpuProver::new(shrink_machine);

        // Get the wrap program, machine, and keys.
        // let wrap_program =
        //     SP1RootVerifier::<InnerConfig, _, _>::build(&shrink_machine, &shrink_vk, false);
        // let wrap_machine = WrapAir::wrap_machine(OuterSC::default());
        // let (wrap_pk, wrap_vk) = wrap_machine.setup(&wrap_program);

        Self {
            recursion_program,
            rec_pk,
            rec_vk,
            deferred_program,
            deferred_pk,
            deferred_vk,
            compress_program,
            compress_pk,
            compress_vk,
            shrink_program,
            shrink_pk,
            shrink_vk,
            core_prover,
            compress_prover,
            shrink_prover,
        }
    }

    /// Creates a proving key and a verifying key for a given RISC-V ELF.
    #[instrument(name = "setup", level = "debug", skip_all)]
    pub fn setup(&self, elf: &[u8]) -> (SP1ProvingKey, SP1VerifyingKey) {
        let program = Program::from(elf);
        let (pk, vk) = self.core_prover.setup(&program);
        let vk = SP1VerifyingKey { vk };
        let pk = SP1ProvingKey {
            pk,
            elf: elf.to_vec(),
            vk: vk.clone(),
        };
        (pk, vk)
    }

    /// Generate a proof of an SP1 program with the specified inputs.
    #[instrument(name = "execute", level = "info", skip_all)]
    pub fn execute(
        elf: &[u8],
        stdin: &SP1Stdin,
        context: SP1Context,
    ) -> Result<(SP1PublicValues, ExecutionReport), ExecutionError> {
        let program = Program::from(elf);
        let opts = SP1CoreOpts::default();
        let mut runtime = Runtime::with_context(program, opts, context);
        runtime.write_vecs(&stdin.buffer);
        for (proof, vkey) in stdin.proofs.iter() {
            runtime.write_proof(proof.clone(), vkey.clone());
        }
        runtime.run_untraced()?;
        Ok((
            SP1PublicValues::from(&runtime.state.public_values_stream),
            runtime.report,
        ))
    }

    /// Generate shard proofs which split up and prove the valid execution of a RISC-V program with
    /// the core prover. Uses the provided context.
    #[instrument(name = "prove_core", level = "info", skip_all)]
    pub fn prove_core<'a>(
        &'a self,
        pk: &SP1ProvingKey,
        stdin: &SP1Stdin,
        opts: SP1ProverOpts,
        mut context: SP1Context<'a>,
    ) -> Result<SP1CoreProof, SP1CoreProverError> {
        context
            .subproof_verifier
            .get_or_insert_with(|| Arc::new(self));
        let program = Program::from(&pk.elf);
        let (proof, public_values_stream) =
            self.prove_core_with_context(program, stdin, opts.core_opts, context)?;
        let public_values = SP1PublicValues::from(&public_values_stream);
        Ok(SP1CoreProof {
            proof: SP1CoreProofData(proof.shard_proofs),
            stdin: stdin.clone(),
            public_values,
        })
    }

    pub fn get_recursion_core_inputs<'a>(
        &'a self,
        vk: &'a StarkVerifyingKey<CoreSC>,
        leaf_challenger: &'a Challenger<CoreSC>,
        shard_proofs: &[ShardProof<CoreSC>],
        batch_size: usize,
        is_complete: bool,
    ) -> Vec<SP1RecursionMemoryLayout<'a, CoreSC, RiscvAir<BabyBear>>> {
        let mut core_inputs = Vec::new();
        let mut reconstruct_challenger = self.core_prover.config().challenger();
        vk.observe_into(&mut reconstruct_challenger);

        // Prepare the inputs for the recursion programs.
        for batch in shard_proofs.chunks(batch_size) {
            let proofs = batch.to_vec();

            let pv = PublicValues::from_vec(proofs[0].public_values.clone());
            core_inputs.push(SP1RecursionMemoryLayout {
                initial_shard: pv.shard,
                current_shard: pv.shard,
                start_pc: pv.start_pc,
                current_pc: pv.start_pc,
                committed_value_digest_arr: pv
                    .committed_value_digest
                    .into_iter()
                    .map(|x| x.0.to_vec())
                    .collect(),
                deferred_proofs_digest_arr: pv.deferred_proofs_digest.to_vec(),
                vk,
                machine: self.core_prover.machine(),
                shard_proofs: proofs,
                leaf_challenger,
                initial_reconstruct_challenger: reconstruct_challenger.clone(),
                is_complete,
                total_core_shards: shard_proofs.len(),
            });

            for proof in batch.iter() {
                reconstruct_challenger.observe(proof.commitment.main_commit);
                reconstruct_challenger.observe_slice(
                    &proof.public_values[0..self.core_prover.machine().num_pv_elts()],
                );
            }
        }

        // Check that the leaf challenger is the same as the reconstruct challenger.
        assert_eq!(
            reconstruct_challenger.sponge_state,
            leaf_challenger.sponge_state
        );
        assert_eq!(
            reconstruct_challenger.input_buffer,
            leaf_challenger.input_buffer
        );
        assert_eq!(
            reconstruct_challenger.output_buffer,
            leaf_challenger.output_buffer
        );
        core_inputs
    }

    pub fn get_recursion_deferred_inputs<'a>(
        &'a self,
        vk: &'a StarkVerifyingKey<CoreSC>,
        leaf_challenger: &'a Challenger<InnerSC>,
        last_proof_pv: &PublicValues<Word<BabyBear>, BabyBear>,
        deferred_proofs: &[ShardProof<InnerSC>],
        batch_size: usize,
        total_core_shards: usize,
    ) -> Vec<SP1DeferredMemoryLayout<'a, InnerSC, RecursionAir<BabyBear, 3>>> {
        // Prepare the inputs for the deferred proofs recursive verification.
        let mut deferred_digest = [Val::<InnerSC>::zero(); DIGEST_SIZE];
        let mut deferred_inputs = Vec::new();

        for batch in deferred_proofs.chunks(batch_size) {
            let proofs = batch.to_vec();

            deferred_inputs.push(SP1DeferredMemoryLayout {
                compress_vk: &self.compress_vk,
                machine: self.compress_prover.machine(),
                proofs,
                start_reconstruct_deferred_digest: deferred_digest.to_vec(),
                is_complete: false,
                sp1_vk: vk,
                sp1_machine: self.core_prover.machine(),
                end_pc: Val::<InnerSC>::zero(),
                end_shard: last_proof_pv.shard + BabyBear::one(),
                leaf_challenger: leaf_challenger.clone(),
                committed_value_digest: last_proof_pv.committed_value_digest.to_vec(),
                deferred_proofs_digest: last_proof_pv.deferred_proofs_digest.to_vec(),
                total_core_shards,
            });

            deferred_digest = Self::hash_deferred_proofs(deferred_digest, batch);
        }
        deferred_inputs
    }

    /// Generate the inputs for the first layer of recursive proofs.
    #[allow(clippy::type_complexity)]
    pub fn get_first_layer_inputs<'a>(
        &'a self,
        vk: &'a SP1VerifyingKey,
        leaf_challenger: &'a Challenger<InnerSC>,
        shard_proofs: &[ShardProof<InnerSC>],
        deferred_proofs: &[ShardProof<InnerSC>],
        batch_size: usize,
    ) -> (
        Vec<SP1RecursionMemoryLayout<'a, InnerSC, RiscvAir<BabyBear>>>,
        Vec<SP1DeferredMemoryLayout<'a, InnerSC, RecursionAir<BabyBear, 3>>>,
    ) {
        let is_complete = shard_proofs.len() == 1 && deferred_proofs.is_empty();
        let core_inputs = self.get_recursion_core_inputs(
            &vk.vk,
            leaf_challenger,
            shard_proofs,
            batch_size,
            is_complete,
        );
        let last_proof_pv =
            PublicValues::from_vec(shard_proofs.last().unwrap().public_values.clone());
        let deferred_inputs = self.get_recursion_deferred_inputs(
            &vk.vk,
            leaf_challenger,
            &last_proof_pv,
            deferred_proofs,
            batch_size,
            shard_proofs.len(),
        );
        (core_inputs, deferred_inputs)
    }

    /// Reduce shards proofs to a single shard proof using the recursion prover.
    #[instrument(name = "compress", level = "info", skip_all)]
    pub fn compress(
        &self,
        vk: &SP1VerifyingKey,
        proof: SP1CoreProof,
        deferred_proofs: Vec<ShardProof<InnerSC>>,
        opts: SP1ProverOpts,
    ) -> Result<SP1ReduceProof<InnerSC>, SP1RecursionProverError> {
        // Set the batch size for the reduction tree.
        let batch_size = 2;

        let shard_proofs = &proof.proof.0;
        let total_core_shards = shard_proofs.len();
        // Get the leaf challenger.
        let mut leaf_challenger = self.core_prover.machine().config().challenger();
        vk.vk.observe_into(&mut leaf_challenger);
        shard_proofs.iter().for_each(|proof| {
            leaf_challenger.observe(proof.commitment.main_commit);
            leaf_challenger
                .observe_slice(&proof.public_values[0..self.core_prover.machine().num_pv_elts()]);
        });

        // Run the recursion and reduce programs.
        let (core_inputs, deferred_inputs) = self.get_first_layer_inputs(
            vk,
            &leaf_challenger,
            shard_proofs,
            &deferred_proofs,
            batch_size,
        );

        let mut reduce_proofs = Vec::new();
        let shard_batch_size = opts.recursion_opts.shard_batch_size;
        for inputs in core_inputs.chunks(shard_batch_size) {
            let proofs = inputs
                .iter()
                .map(|input| {
                    let proof = self.compress_machine_proof(
                        input,
                        &self.recursion_program,
                        &self.rec_pk,
                        opts,
                    );
                    (proof, ReduceProgramType::Core)
                })
                .collect::<Vec<_>>();
            reduce_proofs.extend(proofs);
        }

        // Run the deferred proofs programs.
        for inputs in deferred_inputs.chunks(shard_batch_size) {
            let proofs = inputs
                .iter()
                .map(|input| {
                    let proof = self.compress_machine_proof(
                        input,
                        &self.deferred_program,
                        &self.deferred_pk,
                        opts,
                    );
                    (proof, ReduceProgramType::Deferred)
                })
                .collect::<Vec<_>>();
            reduce_proofs.extend(proofs);
        }

        // Iterate over the recursive proof batches until there is one proof remaining.
        let mut is_complete;
        loop {
            tracing::debug!("Recursive proof layer size: {}", reduce_proofs.len());
            is_complete = reduce_proofs.len() <= batch_size;

            let compress_inputs = reduce_proofs.chunks(batch_size).collect::<Vec<_>>();
            let batched_compress_inputs =
                compress_inputs.chunks(shard_batch_size).collect::<Vec<_>>();
            reduce_proofs = batched_compress_inputs
                .into_iter()
                .flat_map(|batches| {
                    batches
                        .iter()
                        .map(|batch| {
                            let (shard_proofs, kinds) =
                                batch.iter().cloned().unzip::<_, _, Vec<_>, Vec<_>>();

                            let input = SP1ReduceMemoryLayout {
                                compress_vk: &self.compress_vk,
                                recursive_machine: self.compress_prover.machine(),
                                shard_proofs,
                                kinds,
                                is_complete,
                                total_core_shards,
                            };

                            let proof = self.compress_machine_proof(
                                input,
                                &self.compress_program,
                                &self.compress_pk,
                                opts,
                            );
                            (proof, ReduceProgramType::Reduce)
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();

            if reduce_proofs.len() == 1 {
                break;
            }
        }
        debug_assert_eq!(reduce_proofs.len(), 1);
        let reduce_proof = reduce_proofs.pop().unwrap();

        Ok(SP1ReduceProof {
            proof: reduce_proof.0,
        })
    }

    /// Generates a proof using the compress machine and recursive runtime.
    ///
    /// The proof will execute the program and prove it in a single shard.
    pub fn compress_machine_proof(
        &self,
        input: impl Hintable<InnerConfig>,
        program: &RecursionProgram<BabyBear>,
        pk: &StarkProvingKey<InnerSC>,
        opts: SP1ProverOpts,
    ) -> ShardProof<InnerSC> {
        let mut runtime = RecursionRuntime::<Val<InnerSC>, Challenge<InnerSC>, _>::new(
            program,
            self.compress_prover.config().perm.clone(),
        );

        let mut witness_stream = Vec::new();
        witness_stream.extend(input.write());

        runtime.witness_stream = witness_stream.into();
        runtime.run();
        runtime.print_stats();

        let mut recursive_challenger = self.compress_prover.config().challenger();
        self.compress_prover
            .prove(
                pk,
                runtime.record,
                &mut recursive_challenger,
                opts.recursion_opts,
            )
            .shard_proofs
            .pop()
            .unwrap()
    }

    /// Accumulate deferred proofs into a single digest.
    pub fn hash_deferred_proofs(
        prev_digest: [Val<CoreSC>; DIGEST_SIZE],
        deferred_proofs: &[ShardProof<InnerSC>],
    ) -> [Val<CoreSC>; 8] {
        SP1Prover::hash_deferred_proofs(prev_digest, deferred_proofs)
    }

    pub fn prove_core_simple(
        &self,
        runtime: Runtime,
    ) -> Result<MachineProof<CoreSC>, SP1CoreProverError> {
        // Setup the machine.
        let (pk, _) = self.core_prover.setup(runtime.program.as_ref());

        // Prove the program.
        let mut challenger = self.core_prover.config().challenger();
        let proving_start = Instant::now();
        let proof =
            self.core_prover
                .prove(&pk, runtime.record, &mut challenger, SP1CoreOpts::default());
        let proving_duration = proving_start.elapsed().as_millis();
        let nb_bytes = bincode::serialize(&proof).unwrap().len();

        // Print the summary.
        tracing::info!(
            "summary: cycles={}, e2e={}, khz={:.2}, proofSize={}",
            runtime.state.global_clk,
            proving_duration,
            (runtime.state.global_clk as f64 / proving_duration as f64),
            Size::from_bytes(nb_bytes),
        );

        Ok(proof)
    }

    pub fn prove_core_with_context(
        &self,
        program: Program,
        stdin: &SP1Stdin,
        opts: SP1CoreOpts,
        context: SP1Context,
    ) -> Result<(MachineProof<CoreSC>, Vec<u8>), SP1CoreProverError> {
        let proving_start = Instant::now();

        // Execute the program.
        let mut runtime = Runtime::with_context(program.clone(), opts, context);
        runtime.write_vecs(&stdin.buffer);
        for proof in stdin.proofs.iter() {
            runtime.write_proof(proof.0.clone(), proof.1.clone());
        }

        // Setup the machine.
        let (pk, vk) = self.core_prover.setup(runtime.program.as_ref());

        // If we don't need to batch, we can just run the program normally and prove it.
        if opts.shard_batch_size == 0 {
            // Execute the runtime and collect all the events..
            runtime.run().map_err(SP1CoreProverError::ExecutionError)?;

            // If debugging is enabled, we will also debug the constraints.
            #[cfg(debug_assertions)]
            {
                let mut challenger = self.core_prover.config().challenger();
                self.core_prover.machine().debug_constraints(
                    &pk,
                    runtime.record.clone(),
                    &mut challenger,
                );
            }

            // Generate the proof and return the proof and public values.
            let public_values = std::mem::take(&mut runtime.state.public_values_stream);
            let proof = self.prove_core_simple(runtime)?;
            return Ok((proof, public_values));
        }

        // Execute the program, saving checkpoints at the start of every `shard_batch_size` cycle range.
        let mut checkpoints = Vec::new();
        let (public_values_stream, public_values) = loop {
            // Execute the runtime until we reach a checkpoint.
            let (checkpoint, done) = tracing::info_span!("collect_checkpoints")
                .in_scope(|| runtime.execute_state())
                .map_err(SP1CoreProverError::ExecutionError)?;

            // Save the checkpoint to a temp file.
            let mut tempfile = tempfile::tempfile().map_err(SP1CoreProverError::IoError)?;
            let mut writer = std::io::BufWriter::new(&mut tempfile);
            bincode::serialize_into(&mut writer, &checkpoint)
                .map_err(SP1CoreProverError::SerializationError)?;
            writer.flush().map_err(SP1CoreProverError::IoError)?;
            drop(writer);
            tempfile
                .seek(std::io::SeekFrom::Start(0))
                .map_err(SP1CoreProverError::IoError)?;
            checkpoints.push(tempfile);

            // If we've reached the final checkpoint, break out of the loop.
            if done {
                break (
                    std::mem::take(&mut runtime.state.public_values_stream),
                    runtime.record.public_values,
                );
            }
        };

        // For each checkpoint, generate events, shard them, commit shards, and observe in challenger.
        let sharding_config = ShardingConfig::default();
        let mut shard_main_datas = Vec::new();
        let mut challenger = self.core_prover.config().challenger();
        vk.observe_into(&mut challenger);
        for (num, checkpoint_file) in checkpoints.iter_mut().enumerate() {
            let (mut record, _) = tracing::info_span!("commit_checkpoint", num)
                .in_scope(|| trace_checkpoint(program.clone(), checkpoint_file, opts));
            record.public_values = public_values;
            reset_seek(&mut *checkpoint_file);

            // Shard the record into shards.
            let checkpoint_shards = tracing::info_span!("shard")
                .in_scope(|| self.core_prover.machine().shard(record, &sharding_config));

            // Commit to each shard.
            let (commitments, commit_data) = tracing::info_span!("commit")
                .in_scope(|| self.core_prover.commit_shards(&checkpoint_shards));
            shard_main_datas.push(commit_data);

            // Observe the commitments.
            for (commitment, shard) in commitments.into_iter().zip(checkpoint_shards.iter()) {
                challenger.observe(commitment);
                challenger.observe_slice(
                    &shard.public_values::<BabyBear>()[0..self.core_prover.machine().num_pv_elts()],
                );
            }
        }

        // For each checkpoint, generate events and shard again, then prove the shards.
        let mut shard_proofs = Vec::<ShardProof<_>>::new();
        let mut report_aggregate = ExecutionReport::default();
        for (num, mut checkpoint_file) in checkpoints.into_iter().enumerate() {
            let checkpoint_shards = {
                let (mut events, report) = tracing::info_span!("prove_checkpoint", num)
                    .in_scope(|| trace_checkpoint(program.clone(), &checkpoint_file, opts));
                report_aggregate += report;
                events.public_values = public_values;
                reset_seek(&mut checkpoint_file);
                tracing::debug_span!("shard")
                    .in_scope(|| self.core_prover.machine().shard(events, &sharding_config))
            };
            let mut checkpoint_proofs = checkpoint_shards
                .into_iter()
                .map(|shard| {
                    let shard_data = self.core_prover.commit_main(&shard, shard.index() as usize);

                    self.core_prover
                        .prove_shard(&pk, shard_data, &mut challenger.clone())
                        .unwrap()
                })
                .collect::<Vec<_>>();
            shard_proofs.append(&mut checkpoint_proofs);
        }
        // Log some of the `ExecutionReport` information.
        tracing::info!(
            "execution report (totals): total_cycles={}, total_syscall_cycles={}",
            report_aggregate.total_instruction_count(),
            report_aggregate.total_syscall_count()
        );
        // Print the opcode and syscall count tables like `du`:
        // sorted by count (descending) and with the count in the first column.
        tracing::info!("execution report (opcode counts):");
        for line in ExecutionReport::sorted_table_lines(&report_aggregate.opcode_counts) {
            tracing::info!("  {line}");
        }
        tracing::info!("execution report (syscall counts):");
        for line in ExecutionReport::sorted_table_lines(&report_aggregate.syscall_counts) {
            tracing::info!("  {line}");
        }

        let proof = MachineProof { shard_proofs };

        // Print the summary.
        let proving_time = proving_start.elapsed().as_secs_f64();
        tracing::info!(
            "summary: cycles={}, e2e={}, khz={:.2}, proofSize={}",
            runtime.state.global_clk,
            proving_time,
            (runtime.state.global_clk as f64 / proving_time),
            bincode::serialize(&proof).unwrap().len(),
        );

        Ok((proof, public_values_stream))
    }
}

fn trace_checkpoint(
    program: Program,
    file: &File,
    opts: SP1CoreOpts,
) -> (ExecutionRecord, ExecutionReport) {
    let mut reader = std::io::BufReader::new(file);
    let state = bincode::deserialize_from(&mut reader).expect("failed to deserialize state");
    let mut runtime = Runtime::recover(program.clone(), state, opts);
    // We already passed the deferred proof verifier when creating checkpoints, so the proofs were
    // already verified. So here we use a noop verifier to not print any warnings.
    runtime.subproof_verifier = Arc::new(NoOpSubproofVerifier);
    let (events, _) =
        tracing::debug_span!("runtime.trace").in_scope(|| runtime.execute_record().unwrap());
    (events, runtime.report)
}

fn reset_seek(file: &mut File) {
    file.seek(std::io::SeekFrom::Start(0))
        .expect("failed to seek to start of tempfile");
}

#[cfg(test)]
mod tests {

    use super::*;

    use moongate_core::utils::init_tracer;

    use anyhow::Result;
    use sp1_core::io::SP1Stdin;
    use sp1_core::runtime::SP1Context;
    use sp1_core::utils::tests::FIBONACCI_ELF;

    /// Tests an end-to-end workflow of proving a program across the entire proof generation
    /// pipeline.
    ///
    /// Add `FRI_QUERIES`=1 to your environment for faster execution. Should only take a few minutes
    /// on a Mac M2. Note: This test always re-builds the plonk bn254 artifacts, so setting SP1_DEV is
    /// not needed.
    #[test]
    fn test_e2e() -> Result<()> {
        let elf = FIBONACCI_ELF;

        init_tracer();

        let prover = SP1GpuProver::new();

        let cpu_prover = SP1Prover::new();

        tracing::info!("initializing prover");
        let opts = SP1ProverOpts {
            core_opts: SP1CoreOpts {
                shard_size: 1 << 12,
                ..Default::default()
            },
            recursion_opts: SP1CoreOpts::default(),
        };
        let context = SP1Context::default();

        tracing::info!("setup elf");
        let (pk, vk) = prover.setup(elf);

        tracing::info!("prove core");
        let stdin = SP1Stdin::new();
        let core_proof = prover.prove_core(&pk, &stdin, opts, context)?;
        let _public_values = core_proof.public_values.clone();

        tracing::info!("verify core");
        cpu_prover.verify(&core_proof.proof, &vk)?;

        tracing::info!("compress");
        let compressed_proof = prover.compress(&vk, core_proof, vec![], opts)?;

        tracing::info!("verify compressed");
        cpu_prover.verify_compressed(&compressed_proof, &vk)?;

        // tracing::info!("shrink");
        // let shrink_proof = prover.shrink(compressed_proof)?;

        // tracing::info!("verify shrink");
        // prover.verify_shrink(&shrink_proof, &vk)?;

        // tracing::info!("wrap bn254");
        // let wrapped_bn254_proof = prover.wrap_bn254(shrink_proof)?;
        // let bytes = bincode::serialize(&wrapped_bn254_proof).unwrap();

        // // Save the proof.
        // let mut file = File::create("proof-with-pis.bin").unwrap();
        // file.write_all(bytes.as_slice()).unwrap();

        // // Load the proof.
        // let mut file = File::open("proof-with-pis.bin").unwrap();
        // let mut bytes = Vec::new();
        // file.read_to_end(&mut bytes).unwrap();

        // let wrapped_bn254_proof = bincode::deserialize(&bytes).unwrap();

        // tracing::info!("verify wrap bn254");
        // prover.verify_wrap_bn254(&wrapped_bn254_proof, &vk).unwrap();

        // tracing::info!("checking vkey hash babybear");
        // let vk_digest_babybear = wrapped_bn254_proof.sp1_vkey_digest_babybear();
        // assert_eq!(vk_digest_babybear, vk.hash_babybear());

        // tracing::info!("checking vkey hash bn254");
        // let vk_digest_bn254 = wrapped_bn254_proof.sp1_vkey_digest_bn254();
        // assert_eq!(vk_digest_bn254, vk.hash_bn254());

        // tracing::info!("generate plonk bn254 proof");
        // let artifacts_dir =
        //     try_build_plonk_bn254_artifacts_dev(&prover.wrap_vk, &wrapped_bn254_proof.proof);
        // let plonk_bn254_proof = prover.wrap_plonk_bn254(wrapped_bn254_proof, &artifacts_dir);
        // println!("{:?}", plonk_bn254_proof);

        // prover.verify_plonk_bn254(&plonk_bn254_proof, &vk, &public_values, &artifacts_dir)?;

        Ok(())
    }

    // /// Tests an end-to-end workflow of proving a program across the entire proof generation
    // /// pipeline in addition to verifying deferred proofs.
    // #[test]
    // #[serial]
    // fn test_e2e_with_deferred_proofs() -> Result<()> {
    //     setup_logger();

    //     // Test program which proves the Keccak-256 hash of various inputs.
    //     let keccak_elf = include_bytes!("../../tests/keccak256/elf/riscv32im-succinct-zkvm-elf");

    //     // Test program which verifies proofs of a vkey and a list of committed inputs.
    //     let verify_elf = include_bytes!("../../tests/verify-proof/elf/riscv32im-succinct-zkvm-elf");

    //     tracing::info!("initializing prover");
    //     let prover = SP1Prover::new();

    //     tracing::info!("setup keccak elf");
    //     let (keccak_pk, keccak_vk) = prover.setup(keccak_elf);

    //     tracing::info!("setup verify elf");
    //     let (verify_pk, verify_vk) = prover.setup(verify_elf);

    //     tracing::info!("prove subproof 1");
    //     let mut stdin = SP1Stdin::new();
    //     stdin.write(&1usize);
    //     stdin.write(&vec![0u8, 0, 0]);
    //     let deferred_proof_1 = prover.prove_core(&keccak_pk, &stdin)?;
    //     let pv_1 = deferred_proof_1.public_values.as_slice().to_vec().clone();

    //     // Generate a second proof of keccak of various inputs.
    //     tracing::info!("prove subproof 2");
    //     let mut stdin = SP1Stdin::new();
    //     stdin.write(&3usize);
    //     stdin.write(&vec![0u8, 1, 2]);
    //     stdin.write(&vec![2, 3, 4]);
    //     stdin.write(&vec![5, 6, 7]);
    //     let deferred_proof_2 = prover.prove_core(&keccak_pk, &stdin)?;
    //     let pv_2 = deferred_proof_2.public_values.as_slice().to_vec().clone();

    //     // Generate recursive proof of first subproof.
    //     tracing::info!("compress subproof 1");
    //     let deferred_reduce_1 = prover.compress(&keccak_vk, deferred_proof_1, vec![])?;

    //     // Generate recursive proof of second subproof.
    //     tracing::info!("compress subproof 2");
    //     let deferred_reduce_2 = prover.compress(&keccak_vk, deferred_proof_2, vec![])?;

    //     // Run verify program with keccak vkey, subproofs, and their committed values.
    //     let mut stdin = SP1Stdin::new();
    //     let vkey_digest = keccak_vk.hash_babybear();
    //     let vkey_digest: [u32; 8] = vkey_digest
    //         .iter()
    //         .map(|n| n.as_canonical_u32())
    //         .collect::<Vec<_>>()
    //         .try_into()
    //         .unwrap();
    //     stdin.write(&vkey_digest);
    //     stdin.write(&vec![pv_1.clone(), pv_2.clone(), pv_2.clone()]);
    //     stdin.write_proof(deferred_reduce_1.proof.clone(), keccak_vk.vk.clone());
    //     stdin.write_proof(deferred_reduce_2.proof.clone(), keccak_vk.vk.clone());
    //     stdin.write_proof(deferred_reduce_2.proof.clone(), keccak_vk.vk.clone());

    //     tracing::info!("proving verify program (core)");
    //     let verify_proof = prover.prove_core(&verify_pk, &stdin)?;

    //     // Generate recursive proof of verify program
    //     tracing::info!("compress verify program");
    //     let verify_reduce = prover.compress(
    //         &verify_vk,
    //         verify_proof,
    //         vec![
    //             deferred_reduce_1.proof,
    //             deferred_reduce_2.proof.clone(),
    //             deferred_reduce_2.proof,
    //         ],
    //     )?;
    //     let reduce_pv: &RecursionPublicValues<_> =
    //         verify_reduce.proof.public_values.as_slice().borrow();
    //     println!("deferred_hash: {:?}", reduce_pv.deferred_proofs_digest);
    //     println!("complete: {:?}", reduce_pv.is_complete);

    //     tracing::info!("verify verify program");
    //     prover.verify_compressed(&verify_reduce, &verify_vk)?;

    //     Ok(())
    // }
}
