#include "utils.hpp"

#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>

void write_string_to_int_arr(uint32_t* a, const std::string& s) {
	size_t char_idx = s.length() - 1;
	size_t uints_per_field_elem = 0;
	while (true) {
		std::string this_index_int;
		while (s[char_idx] != 'x' && this_index_int.length() < 8) {
			this_index_int = s[char_idx] + this_index_int;
			--char_idx;
		}

		if (this_index_int.length() == 0) {
			break;
		}

		a[uints_per_field_elem] = std::stol(this_index_int, (size_t*)nullptr, 16);

		++uints_per_field_elem;
	}
}

std::string int_to_hex(unsigned int i) {
	std::stringstream stream;
	stream << std::hex << i;
	return stream.str();
}