#pragma once

#include "ggml.h"
#include "ggml-backend.h"


#ifdef  __cplusplus
extern "C" {
#endif

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_npu_init(void);

GGML_BACKEND_API bool ggml_backend_is_npu(ggml_backend_t backend);

// number of threads used for conversion to float
// for openblas and blis, this will also set the number of threads used for blas operations
GGML_BACKEND_API void ggml_backend_npu_set_n_threads(ggml_backend_t backend_npu, int n_threads);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_npu_reg(void);


#ifdef  __cplusplus
}
#endif
