#if !defined(TWODIMS) && !defined(ONEDIM)
#define  NUMDIMS 3		/*!< For 3D-normalized kernel */
#define  NORM_COEFF      4.188790204786	/*!< Coefficient for kernel normalization. Note:  4.0/3 * PI = 4.188790204786 */
#else
#ifndef  ONEDIM
#define  NUMDIMS 2		/*!< For 2D-normalized kernel */
#define  NORM_COEFF      M_PI	/*!< Coefficient for kernel normalization. */
#else
#define  NUMDIMS 1             /*!< For 1D-normalized kernel */
#define  NORM_COEFF      2.0
#endif
#endif

typedef struct {
    double H;
    double HH;
    double Hinv; /* convert from r to u*/
    int type;
    double support;
    char * name;
    /* private: */
    double Wknorm;
    double dWknorm;
} DensityKernel;

double
density_kernel_desnumngb(DensityKernel * kernel, double eta);
void
density_kernel_init(DensityKernel * kernel, double H);
void
density_kernel_init_with_type(DensityKernel * kernel, int type, double H);
double
density_kernel_wk(DensityKernel * kernel, double u);
double
density_kernel_dwk(DensityKernel * kernel, double u);
double
density_kernel_volume(DensityKernel * kernel);

static inline double
density_kernel_dW(DensityKernel * kernel, double u, double wk, double dwk)
{
    return - (NUMDIMS * kernel->Hinv * wk + u * dwk);
}

static inline double
dotproduct(double v1[3], double v2[3])
{
    double r =0;
    int d;
    for(d = 0; d < 3; d ++) {
        r += v1[d] * v2[d];
    }
    return r;
}

static inline void crossproduct(double v1[3], double v2[3], double out[3])
{
    static int D2[3] = {1, 2, 0};
    static int D3[3] = {2, 0, 1};

    int d1, d2, d3;

    for(d1 = 0; d1 < 3; d1++)
    {
        d2 = D2[d1];
        d3 = D3[d1];

        out[d1] = (v1[d2] * v2[d3] -  v2[d2] * v1[d3]);
    }
}
