 Fastest possible x86 implementation of popcount (aka population count,
 aka Hamming weight, aka counting the number of set bits in a bit array),
 particularly for larger datasets where the performance really matters.

 Intel microarchitectures since Nehalem and AMD since K10 support
 the `POPCNT` x86 instruction, which can in modern archs sustain
 1 or even 2 64-bit popcounts per cycle. This is optimal
 if you don't have AVX2 or above. Just partially unroll an `_mm_popcnt_u64()`
 loop and you're done.

 NOTE: if you do this, be advised that Intel architectures until Cannon Lake
 have a false dependency on the destination register in the popcnt instruction.
 This means that performance, on most Intel archs, will be 1/3 of theoretical
 if you're not careful, as the popcnt instruction has a throughput of 1 and
 a latency of 3, so if it can't schedule independent popcounts independently,
 it will only complete one every 3 cycles instead of every 1.
 Clang and gcc are aware of this deficiency and automatically account for it,
 but MSVC isn't (of course), so in order to get decent performance, you
 MUST implement it in assembly, something like this:

```
 xor eax, eax
 add rcx, rdx
 neg rdx
 jz LOOP_END
 ALIGN 16
 LOOP_TOP:
 popcnt r8, qword ptr [rcx + rdx + 0]
 add eax, r8d
 popcnt r9, qword ptr [rcx + rdx + 8]
 add eax, r9d
 popcnt r10, qword ptr [rcx + rdx + 16]
 add eax, r10d
 popcnt r11, qword ptr [rcx + rdx + 24]
 add eax, r11d
 add rdx, 32
 jnz LOOP_TOP
 LOOP_END:
 ret
 ```

 In which you send at least 3 popcounts to separate destination registers
 to avoid false dependency troubles.

 Obviously, you can flesh this out to handle bytes < 32, etc.

 However, it is possible to do *MUCH* better if you have `AVX2` or above.

 If you have `AVX512VPOPCNTDQ`, it's trivial: just use `_mm512_popcnt_epi64()`,
 or `_mm256_popcnt_epi64()` if you also have `AVX512VL` and don't want to
 use 512-wides.

 If you just have `AVX2` but no higher (quite common), use the implementation
 below, which combines two tricks:

 The first trick is to popcount vectors by splitting the data into nybbles
 and using them as indices into a shuffle LUT to get bit counts,
 effectively implementing a slightly slower version of `_mm256_popcnt_epi64()`.
 Clang will actually autogenerate something like this from naive popcount
 loops in some cases, which is pretty cool. However, it does *NOT* produce
 optimal performance, because it does not know about trick 2:

 The second trick is to make up for the fact that this software implementation
 of `_mm256_popcnt_epi64()` is not super fast, so you can benefit from doing
 it as little as possible, if there is a way to still count all the bits.
 Well, there is: chain the data into a Harley-Seal CSA (carry-save adder).
 Then you need only call the vector popcount about 1/16 as often.

 Combining these tricks produces the best possible performance, and compilers
 will not do this for you (yet).

 Compiles slightly better under Clang than gcc, and slightly better under gcc
 than MSVC. Go assembly if you have to.
