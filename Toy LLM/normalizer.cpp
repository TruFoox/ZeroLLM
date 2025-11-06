#include <windows.h>
#include <string>
#include <stdexcept>
#include <iostream>



// NOT MY CODE: I REALLY DIDNT WANT TO DEAL WITH THIS SHIT

std::string normalize(const std::string& input) {
    if (input.empty()) return input;

    // Step 1: convert UTF-8 to UTF-16
    int utf16Size = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, input.c_str(), static_cast<int>(input.size()), nullptr, 0);
    if (utf16Size == 0) {
        std::cerr << "Warning: UTF-8 to UTF-16 conversion failed. Returning original string.\n";
        return input; // fallback: return original string
    }

    std::wstring wide(utf16Size, 0);
    if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, input.c_str(), static_cast<int>(input.size()), &wide[0], utf16Size) == 0) {
        std::cerr << "Warning: UTF-8 to UTF-16 conversion failed on second call. Returning original string.\n";
        return input;
    }

    // Step 2: normalize UTF-16 string
    int normSize = NormalizeString(NormalizationC, wide.c_str(), utf16Size, nullptr, 0);
    if (normSize == 0) {
        std::cerr << "Warning: Normalization failed. Returning original string.\n";
        return input;
    }

    std::wstring normalized(normSize, 0);
    if (NormalizeString(NormalizationC, wide.c_str(), utf16Size, &normalized[0], normSize) == 0) {
        std::cerr << "Warning: Normalization failed on second call. Returning original string.\n";
        return input;
    }

    // Step 3: convert back to UTF-8
    int utf8Size = WideCharToMultiByte(CP_UTF8, 0, normalized.c_str(), normSize, nullptr, 0, nullptr, nullptr);
    if (utf8Size == 0) {
        std::cerr << "Warning: UTF-16 to UTF-8 conversion failed. Returning original string.\n";
        return input;
    }

    std::string output(utf8Size, 0);
    if (WideCharToMultiByte(CP_UTF8, 0, normalized.c_str(), normSize, &output[0], utf8Size, nullptr, nullptr) == 0) {
        std::cerr << "Warning: UTF-16 to UTF-8 conversion failed on second call. Returning original string.\n";
        return input;
    }

    // Remove potential null terminator added by WideCharToMultiByte
    if (!output.empty() && output.back() == '\0') output.pop_back();

    return output;
}
