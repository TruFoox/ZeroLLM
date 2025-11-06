#include <windows.h>
#include <string>
#include <iostream>
#include <algorithm>

/* 
I put this in a seperate file because I did not make it.

Tokenization is the worst part of making an LLM, in my opinion, and I REALLY did not want want to deal with this normalization bullshit
*/

std::string normalize(const std::string& input) {
    if (input.empty()) return input;

    // Step 1: Convert UTF-8 to UTF-16
    int utf16Size = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, input.c_str(), (int)input.size(), nullptr, 0);
    std::wstring wide;
    if (utf16Size == 0) {
        // Replace invalid UTF-8 bytes with '?'
        wide.resize(input.size());
        for (size_t i = 0; i < input.size(); ++i)
            wide[i] = static_cast<unsigned char>(input[i]) < 128 ? input[i] : L'?';
    }
    else {
        wide.resize(utf16Size);
        if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, input.c_str(), (int)input.size(), &wide[0], utf16Size) == 0) {
            wide.clear();
            wide.resize(input.size());
            for (size_t i = 0; i < input.size(); ++i)
                wide[i] = static_cast<unsigned char>(input[i]) < 128 ? input[i] : L'?';
        }
    }

    // Step 2: Normalize UTF-16 (NFC)
    int normSize = NormalizeString(NormalizationC, wide.c_str(), (int)wide.size(), nullptr, 0);
    std::wstring normalized(normSize, 0);
    if (normSize > 0) {
        if (NormalizeString(NormalizationC, wide.c_str(), (int)wide.size(), &normalized[0], normSize) == 0) {
            normalized = wide; // fallback
        }
    }
    else {
        normalized = wide; // fallback
    }

    // Step 3: Remove formatting in UTF-16
    std::wstring cleaned;
    for (wchar_t wc : normalized) {
        if (wc == L'\n' || wc == L'\r') {
            cleaned += L' ';
        }
        else if (wc != L'*' && wc != L'_' && wc != L'\t' && wc != L'"' && wc != L'\'') {
            cleaned += wc;
        }
    }

    // Step 4: Convert back to UTF-8
    int utf8Size = WideCharToMultiByte(CP_UTF8, 0, cleaned.c_str(), (int)cleaned.size(), nullptr, 0, nullptr, nullptr);
    if (utf8Size == 0) return ""; // fallback

    std::string output(utf8Size, 0);
    if (WideCharToMultiByte(CP_UTF8, 0, cleaned.c_str(), (int)cleaned.size(), &output[0], utf8Size, nullptr, nullptr) == 0)
        return "";

    return output;
}
