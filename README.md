# moongate

# Setup

The dependencies set in this repo assume the `sp1` and `Plonky3` dependencies are local. This is found to be convenient as there is often a need to make things public or make other small changes. To set up for local developement, the user should set up local copies of Plonky3 and SP1.

### Plonky3 Dependency

In the parent directory, 

```bash
git clone https://github.com/Plonky3/Plonky3.git
cd Plonky3
git checkout sp1-v2-pub
```

### Sp1 Dependency
In the parent directory, do the follwing:

Clone `sp1` repository:
```bash
git clone https://github.com/succinctlabs/sp1.git
cd sp1
git checkout dev
```
then move the plonky3 depencies to the ones for local development.
