#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>

#include "./constants.hpp"

class VarTable {
private:
	std::map<uint32_t *, uint32_t> mem_address_to_var_name;
	uint32_t global_var_number = 0;
	std::stringstream program;

	uint32_t address_to_name(uint32_t *address) {
		if (mem_address_to_var_name[address]) {
			return mem_address_to_var_name[address];
		} else {
			++(global_var_number);
			mem_address_to_var_name[address] = global_var_number;
			return global_var_number;
		}
	}

	void write_and(uint32_t *a, uint32_t *b, uint32_t *destination) {
		*destination ^= *a & *b;
		program << "v" << address_to_name(destination) << " ^= v" << address_to_name(a) << " & v" << address_to_name(b)
				<< ";" << std::endl;
	}

	void write_and_without_xor(uint32_t *a, uint32_t *b, uint32_t *destination) {
		*destination = *a & *b;
		program << "v" << address_to_name(destination) << " = v" << address_to_name(a) << " & v" << address_to_name(b)
				<< ";" << std::endl;
	}

	void copy_without_xor(uint32_t *from, uint32_t *to, uint32_t width) {
		for (uint32_t i = 0; i < width; ++i) {
			to[i] = from[i];

			program << "v" << address_to_name(to + i) << " = v" << address_to_name(from + i) << ";" << std::endl;
		}
	}

	void copy(uint32_t *from, uint32_t *to, uint32_t width) {
		for (uint32_t i = 0; i < width; ++i) {
			to[i] ^= from[i];
			program << "v" << address_to_name(to + i) << " ^= v" << address_to_name(from + i) << ";" << std::endl;
		}
	}

	void xor_halves(uint32_t *from, uint32_t *to, uint32_t half_width) {
		for (uint32_t i = 0; i < half_width; ++i) {
			to[i] = from[i + half_width] ^ from[i];

			program << "v" << address_to_name(to + i) << " = v" << address_to_name(from + i) << " ^ v"
					<< address_to_name(from + i + half_width) << ";" << std::endl;
		}
	}

	void multiply_alpha(
		uint32_t *field_element, uint32_t *destination, uint32_t field_element_width, bool writing_to_zeros
	) {
		if (field_element_width == 1) {
			if (writing_to_zeros) {
				copy_without_xor(field_element, destination, 1);
			} else {
				copy(field_element, destination, 1);
			}
			return;
		}

		uint32_t half_width = field_element_width >> 1;

		if (writing_to_zeros) {
			copy_without_xor(field_element + half_width, destination, half_width);
			copy_without_xor(field_element, destination + half_width, half_width);
		} else {
			copy(field_element + half_width, destination, half_width);
			copy(field_element, destination + half_width, half_width);
		}
		multiply_alpha(field_element + half_width, destination + half_width, half_width, false);
	}

	void multiply(
		uint32_t *field_element_a,
		uint32_t *field_element_b,
		uint32_t *destination,
		uint32_t field_element_width,
		bool writing_to_zeros,
		uint32_t *blank_memory_pointer,
		uint32_t &bmp_slot_index
	) {
		if (field_element_width == 1) {
			if (writing_to_zeros) {
				write_and_without_xor(field_element_a, field_element_b, destination);
			} else {
				write_and(field_element_a, field_element_b, destination);
			}
			return;
		}

		// field_element is the start of the low half
		uint32_t half_width = field_element_width >> 1;

		uint32_t *z2_z0 = blank_memory_pointer + bmp_slot_index;

		bmp_slot_index += half_width;

		// Load z2 = a_hi*b_hi into the lower half of result
		multiply(
			field_element_a + half_width,
			field_element_b + half_width,
			z2_z0,
			half_width,
			true,
			blank_memory_pointer,
			bmp_slot_index
		);

		// Load z2a = a_hi*b_hi*alpha into the upper half of result
		multiply_alpha(z2_z0, destination + half_width, half_width, writing_to_zeros);

		uint32_t *xored_half_a = blank_memory_pointer + bmp_slot_index;
		bmp_slot_index += half_width;

		uint32_t *xored_half_b = blank_memory_pointer + bmp_slot_index;
		bmp_slot_index += half_width;

		xor_halves(field_element_a, xored_half_a, half_width);
		xor_halves(field_element_b, xored_half_b, half_width);

		// Load z0 = a_lo*b_lo into the lower half of result
		multiply(field_element_a, field_element_b, z2_z0, half_width, false, blank_memory_pointer, bmp_slot_index);

		if (writing_to_zeros) {
			copy_without_xor(z2_z0, destination, half_width);
		} else {
			copy(z2_z0, destination, half_width);
		}

		copy(z2_z0, destination + half_width, half_width);

		// Load z1 = (a_hi+a_lo)*(b_hi+b_lo) into the upper half of result
		multiply(
			xored_half_a,
			xored_half_b,
			destination + half_width,
			half_width,
			false,
			blank_memory_pointer,
			bmp_slot_index
		);
	}

public:
	void multiply_and_generate(uint32_t *field_element_a, uint32_t *field_element_b, uint32_t *destination) {
		for (uint32_t i = 0; i < BITS_WIDTH; ++i) {
			destination[i] = 0;
		}

		uint32_t blank_memory_pointer[BLANK_MEMORY_SIZE] = {};

		uint32_t bmp_slot_index = 0;

		multiply(field_element_a, field_element_b, destination, BITS_WIDTH, true, blank_memory_pointer, bmp_slot_index);

		std::string program_str(std::istreambuf_iterator<char>(program), {});

		std::stringstream input_reads;

		std::stringstream output_writes;

		for (uint32_t i = 0; i < BITS_WIDTH; ++i) {
			uint32_t a_varname = mem_address_to_var_name[field_element_a + i];
			uint32_t b_varname = mem_address_to_var_name[field_element_b + i];
			uint32_t dst_varname = mem_address_to_var_name[destination + i];

			input_reads << "v" << a_varname << " = field_element_a[" << i << "];" << std::endl;
			input_reads << "v" << b_varname << " = field_element_b[" << i << "];" << std::endl;
			output_writes << "destination[" << i << "] = v" << dst_varname << ";" << std::endl;
		}

		std::string input_reads_str(std::istreambuf_iterator<char>(input_reads), {});

		std::string output_writes_str(std::istreambuf_iterator<char>(output_writes), {});

		std::stringstream declarations;

		uint32_t highest_variable_num = global_var_number;

		for (uint32_t i = 1; i <= highest_variable_num; ++i) {
			declarations << "uint32_t v" << i << ";" << std::endl;
		}

		std::string declarations_str(std::istreambuf_iterator<char>(declarations), {});

		std::string file_name = "./unrolled/binary_tower_unrolled";
		file_name += std::to_string(TOWER_HEIGHT);
		file_name += ".cu";

		std::fstream f(file_name, std::ios::out);

		f << "//This file is auto generated by multiply_and_generate_circuit.cpp" << std::endl;

		f << "#include <cstdint>" << std::endl;

		f << "#include \"binary_tower_unrolled.cuh\"" << std::endl;

		f << "template<>" << std::endl;

		f << "__host__ __device__ void multiply_unrolled<" << TOWER_HEIGHT
		  << ">(const uint32_t *field_element_a, const uint32_t *field_element_b, uint32_t *destination){" << std::endl;

		f << declarations_str << std::endl;

		f << input_reads_str << std::endl;

		f << program_str << std::endl;

		f << output_writes_str << std::endl;

		f << "}" << std::endl;
	}
};

int main() {
	uint32_t a[BITS_WIDTH];  // i_7 = <arr[0] bit 7, arr[1] bit 7, arr[2] bit 7...>

	uint32_t b[BITS_WIDTH];

	uint32_t mult_result[BITS_WIDTH];

	VarTable v;

	v.multiply_and_generate(a, b, mult_result);

	std::cout << "Generated multiplication circuit of tower height " << TOWER_HEIGHT << std::endl;

	return 0;
}