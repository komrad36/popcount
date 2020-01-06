/*******************************************************************
*
*	Author: Kareem Omar
*	kareem.h.omar@gmail.com
*	https://github.com/komrad36
*
*	Last updated Jan 6, 2020
*******************************************************************/
//
// Fastest possible x86 implementation of popcount (aka population count,
// aka Hamming weight, aka counting the number of set bits in a bit array),
// particularly for larger datasets where the performance really matters.
//
// Intel microarchitectures since Nehalem and AMD since K10 support
// the POPCNT x86 instruction, which can in modern archs sustain
// 1 or even 2 64-bit popcounts per cycle. This is optimal
// if you don't have AVX2 or above. Just partially unroll an _mm_popcnt_u64()
// loop and you're done.
//
// NOTE: if you do this, be advised that Intel architectures until Cannon Lake
// have a false dependency on the destination register in the popcnt instruction.
// This means that performance, on most Intel archs, will be 1/3 of theoretical
// if you're not careful, as the popcnt instruction has a throughput of 1 and
// a latency of 3, so if it can't schedule independent popcounts independently,
// it will only complete one every 3 cycles instead of every 1.
// Clang and gcc are aware of this deficiency and automatically account for it,
// but MSVC isn't (of course), so in order to get decent performance, you
// MUST implement it in assembly, something like this:
//
// xor eax, eax
// add rcx, rdx
// neg rdx
// jz LOOP_END
// ALIGN 16
// LOOP_TOP:
// popcnt r8, qword ptr [rcx + rdx + 0]
// add eax, r8d
// popcnt r9, qword ptr [rcx + rdx + 8]
// add eax, r9d
// popcnt r10, qword ptr [rcx + rdx + 16]
// add eax, r10d
// popcnt r11, qword ptr [rcx + rdx + 24]
// add eax, r11d
// add rdx, 32
// jnz LOOP_TOP
// LOOP_END:
// ret
//
// In which you send at least 3 popcounts to separate destination registers
// to avoid false dependency troubles.
//
// Obviously, you can flesh this out to handle bytes < 32, etc.
//
// However, it is possible to do *MUCH* better if you have AVX2 or above.
//
// If you have AVX512VPOPCNTDQ, it's trivial: just use _mm512_popcnt_epi64(),
// or _mm256_popcnt_epi64() if you also have AVX512VL and don't want to
// use 512-wides.
//
// If you just have AVX2 but no higher (quite common), use the implementation
// below, which combines two tricks:
//
// The first trick is to popcount vectors by splitting the data into nybbles
// and using them as indices into a shuffle LUT to get bit counts,
// effectively implementing a slightly slower version of _mm256_popcnt_epi64().
// Clang will actually autogenerate something like this from naive popcount
// loops in some cases, which is pretty cool. However, it does *NOT* produce
// optimal performance, because it does not know about trick 2:
//
// The second trick is to make up for the fact that this software implementation
// of _mm256_popcnt_epi64() is not super fast, so you can benefit from doing
// it as little as possible, if there is a way to still count all the bits.
// Well, there is: chain the data into a Harley-Seal CSA (carry-save adder).
// Then you need only call the vector popcount about 1/16 as often.
//
// Combining these tricks produces the best possible performance, and compilers
// will not do this for you (yet).
//
// Compiles slightly better under Clang than gcc, and slightly better under gcc
// than MSVC. Go assembly if you have to.
//

#pragma once

#include <cstdint>
#include <immintrin.h>

using I64 = int64_t;
using U64 = uint64_t;

static inline __m256i mm256_popcnt_epi64(const __m256i& p) {
	const __m256i lut = _mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
	const __m256i m = _mm256_set1_epi8(15);

	const __m256i L = _mm256_and_si256(p, m);
	const __m256i H = _mm256_and_si256(_mm256_srli_epi32(p, 4), m);
	const __m256i cL = _mm256_shuffle_epi8(lut, L);
	const __m256i cH = _mm256_shuffle_epi8(lut, H);
	return _mm256_sad_epu8(_mm256_add_epi8(cL, cH), _mm256_setzero_si256());
}

static inline void CSA(__m256i& h, __m256i& l, const __m256i& a, const __m256i& b, const __m256i& c) {
	const __m256i u = _mm256_xor_si256(a, b);
	h = _mm256_or_si256(_mm256_and_si256(a, b), _mm256_and_si256(u, c));
	l = _mm256_xor_si256(u, c);
}

// requires bytes to be a multiple of 8
U64 t9(const U64* const __restrict data, I64 bytes) {
	const uintptr_t p = (uintptr_t)data;
	__m256i ya = _mm256_setzero_si256();
	I64 i = 0;

	if (bytes >= 1024) {
		__m256i ones = _mm256_setzero_si256();
		__m256i twos = _mm256_setzero_si256();
		__m256i fours = _mm256_setzero_si256();
		__m256i eights = _mm256_setzero_si256();
		__m256i sixteens = _mm256_setzero_si256();
		__m256i twosA, twosB, foursA, foursB, eightsA, eightsB;

		for (; i < bytes - 511; i += 512) {
			CSA(twosA, ones, ones, _mm256_loadu_si256((const __m256i*)(p + i)), _mm256_loadu_si256((const __m256i*)(p + i + 1 * 32)));
			CSA(twosB, ones, ones, _mm256_loadu_si256((const __m256i*)(p + i + 2 * 32)), _mm256_loadu_si256((const __m256i*)(p + i + 3 * 32)));
			CSA(foursA, twos, twos, twosA, twosB);
			CSA(twosA, ones, ones, _mm256_loadu_si256((const __m256i*)(p + i + 4 * 32)), _mm256_loadu_si256((const __m256i*)(p + i + 5 * 32)));
			CSA(twosB, ones, ones, _mm256_loadu_si256((const __m256i*)(p + i + 6 * 32)), _mm256_loadu_si256((const __m256i*)(p + i + 7 * 32)));
			CSA(foursB, twos, twos, twosA, twosB);
			CSA(eightsA, fours, fours, foursA, foursB);
			CSA(twosA, ones, ones, _mm256_loadu_si256((const __m256i*)(p + i + 8 * 32)), _mm256_loadu_si256((const __m256i*)(p + i + 9 * 32)));
			CSA(twosB, ones, ones, _mm256_loadu_si256((const __m256i*)(p + i + 10 * 32)), _mm256_loadu_si256((const __m256i*)(p + i + 11 * 32)));
			CSA(foursA, twos, twos, twosA, twosB);
			CSA(twosA, ones, ones, _mm256_loadu_si256((const __m256i*)(p + i + 12 * 32)), _mm256_loadu_si256((const __m256i*)(p + i + 13 * 32)));
			CSA(twosB, ones, ones, _mm256_loadu_si256((const __m256i*)(p + i + 14 * 32)), _mm256_loadu_si256((const __m256i*)(p + i + 15 * 32)));
			CSA(foursB, twos, twos, twosA, twosB);
			CSA(eightsB, fours, fours, foursA, foursB);
			CSA(sixteens, eights, eights, eightsA, eightsB);

			ya = _mm256_add_epi64(ya, mm256_popcnt_epi64(sixteens));
		}

		ya = _mm256_slli_epi64(ya, 4);
		ya = _mm256_add_epi64(ya, _mm256_slli_epi64(mm256_popcnt_epi64(eights), 3));
		ya = _mm256_add_epi64(ya, _mm256_slli_epi64(mm256_popcnt_epi64(fours), 2));
		ya = _mm256_add_epi64(ya, _mm256_slli_epi64(mm256_popcnt_epi64(twos), 1));
		ya = _mm256_add_epi64(ya, mm256_popcnt_epi64(ones));
	}

	for (; i < bytes - 63; i += 64) {
		ya = _mm256_add_epi64(ya, mm256_popcnt_epi64(_mm256_loadu_si256((const __m256i*)(p + i))));
		ya = _mm256_add_epi64(ya, mm256_popcnt_epi64(_mm256_loadu_si256((const __m256i*)(p + i + 32))));
	}

	for (; i < bytes - 31; i += 32) {
		ya = _mm256_add_epi64(ya, mm256_popcnt_epi64(_mm256_loadu_si256((const __m256i*)(p + i))));
	}

	const __m128i xa = _mm_add_epi64(_mm256_castsi256_si128(ya), _mm256_extracti128_si256(ya, 1));
	U64 a = (U64)_mm_cvtsi128_si64(_mm_add_epi64(xa, _mm_shuffle_epi32(xa, 78)));

	for (; i < bytes - 7; i += 8) {
		a += _mm_popcnt_u64(*(const U64*)(p + i));
	}

	return a;
}
