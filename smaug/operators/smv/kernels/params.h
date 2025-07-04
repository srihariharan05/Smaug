#ifndef _OPERATORS_SMV_KERNELS_PARAMS_H_
#define _OPERATORS_SMV_KERNELS_PARAMS_H_

#ifndef VECTOR_SIZE
#define VECTOR_SIZE 8
#elif VECTOR_SIZE != 8
#error "Existing VECTOR_SIZE is incompatible with SMV!"
#endif

#define ACC_256_GOPS    1
#define ACC_1_TOPS      0
#define ACC_4_TOPS      0
#define ACC_8_TOPS      0
#define ACC_32_TOPS     0

#if ACC_256_GOPS
    #define NUM_MACC_INSTS 4
    #define NUM_PE_INSTS 8
#elif ACC_1_TOPS
    #define NUM_MACC_INSTS 4
    #define NUM_PE_INSTS 32
#elif ACC_4_TOPS   
    #define NUM_MACC_INSTS  16
    #define NUM_PE_INSTS    32
#elif ACC_32_TOPS
    #define NUM_MACC_INSTS 16
    #define NUM_PE_INSTS 256
#endif

#define DATA_PE_ALIGNMENT (NUM_MACC_INSTS)*(VECTOR_SIZE)
#define RESULT_ALIGNMENT  (NUM_PE_INSTS/VECTOR_SIZE)
#endif
