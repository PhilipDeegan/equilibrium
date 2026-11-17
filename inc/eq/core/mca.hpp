#ifndef EQUILIBRIUM_CORE_MCA_HPP
#define EQUILIBRIUM_CORE_MCA_HPP

#if defined(EQ_LLVM_MCA)
#define EQ_MCA_BEGIN(name) __asm volatile("# LLVM-MCA-BEGIN " name)
#define EQ_MCA_END(name) __asm volatile("# LLVM-MCA-END " name)
#else
#define EQ_MCA_BEGIN(name)
#define EQ_MCA_END(name)
#endif

#endif /* EQUILIBRIUM_CORE_MCA_HPP */
