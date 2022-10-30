#include <mpi.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>

#include "utils.h"

#include "partmanager.h"
#include "forcetree.h"
#include "petapm.h"
#include "domain.h"
#include "walltime.h"
#include "gravity.h"

#include "cosmology.h"
#include "neutrinos_lra.h"

static int pm_mark_region_for_node(int startno, int rid, int * RegionInd, const ForceTree * tt);
static void convert_node_to_region(PetaPM * pm, PetaPMRegion * r, struct NODE * Nodes);

static int hybrid_nu_gravpm_is_active(int i);
static void potential_transfer(PetaPM * pm, int64_t k2, int kpos[3], pfft_complex * value);
static void compute_neutrino_power(PetaPM * pm);
static void force_x_transfer(PetaPM * pm, int64_t k2, int kpos[3], pfft_complex * value);
static void force_y_transfer(PetaPM * pm, int64_t k2, int kpos[3], pfft_complex * value);
static void force_z_transfer(PetaPM * pm, int64_t k2, int kpos[3], pfft_complex * value);
static void readout_potential(PetaPM * pm, int i, double * mesh, double weight);
static void readout_force_x(PetaPM * pm, int i, double * mesh, double weight);
static void readout_force_y(PetaPM * pm, int i, double * mesh, double weight);
static void readout_force_z(PetaPM * pm, int i, double * mesh, double weight);
static PetaPMFunctions functions [] =
{
    {"Potential", NULL, readout_potential},
    {"ForceX", force_x_transfer, readout_force_x},
    {"ForceY", force_y_transfer, readout_force_y},
    {"ForceZ", force_z_transfer, readout_force_z},
    {NULL, NULL, NULL},
};

static PetaPMRegion * _prepare(PetaPM * pm, PetaPMParticleStruct * pstruct, void * userdata, int * Nregions);

static struct gravpm_params
{
    double Time;
    double TimeIC;
    Cosmology * CP;
    int FastParticleType;
    double UnitLength_in_cm;
    double Xmax[3],Xmin[3];
    double BoxSize;
} GravPM;


/* Calculate the box size based on particle positions*/
void  
gravpm_set_lbox_nonperiodic(void) {
    int NumPart = PartManager->NumPart;
    int k;
    int i;
    double Xmin[3] = {1.0e30, 1.0e30, 1.0e30};
    double Xmax[3] = {-1.0e30, -1.0e30, -1.0e30};
    
    #pragma omp parallel for
    for(i = 0; i < NumPart; i ++) {
        for(k = 0; k < 3; k ++) {
            if(Xmin[k] > P[i].Pos[k])
            Xmin[k] = P[i].Pos[k];
            if(Xmax[k] < P[i].Pos[k])
            Xmax[k] = P[i].Pos[k];
        }
    }
    
    MPI_Allreduce(MPI_IN_PLACE, &Xmin, 3, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &Xmax, 3, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    GravPM.BoxSize = Xmax[0] - Xmin[0];
    GravPM.BoxSize = dmax(GravPM.BoxSize, Xmax[1] - Xmin[1]);
    GravPM.BoxSize = dmax(GravPM.BoxSize, Xmax[2] - Xmin[2]);
    MPI_Barrier(MPI_COMM_WORLD);
    /* redefine xmax by boxsize */
    for(i = 0; i < 3; i++) {
        Xmax[i] = Xmin[i] + GravPM.BoxSize;
        GravPM.Xmin[i] = Xmin[i];
        GravPM.Xmax[i] = Xmax[i];
    }
    GravPM.BoxSize *= 2.;
}


void
gravpm_init_nonperiodic(PetaPM * pm, double BoxSize, double Asmth, int Nmesh, double G, int NonPeriodic) {
/* does not matter if we use any boxsize in initialization because it's only used to inform pm->BoxSize
  we will be okay if we substitute pm->BoxSize to GravPM->BoxSize properly */
    gravpm_set_lbox_nonperiodic();
    petapm_init(pm, GravPM.BoxSize, GravPM->Xmin, Asmth, 2*Nmesh, G, NonPeriodic, MPI_COMM_WORLD);
}

/* Computes the gravitational force on the PM grid
 * and saves the total matter power spectrum.
 * Parameters: Cosmology, Time, UnitLength_in_cm and PowerOutputDir are used by the power spectrum output code.
 * TimeIC and FastParticleType are used by the massive neutrino code. FastParticleType denotes possibly inactive particles.*/
void
gravpm_force_nonperiodic(PetaPM * pm, ForceTree * tree, Cosmology * CP, double Time, double UnitLength_in_cm, const char * PowerOutputDir, double TimeIC, int FastParticleType) {
    PetaPMParticleStruct pstruct = {
        P,
        sizeof(P[0]),
        (char*) &P[0].Pos[0]  - (char*) P,
        (char*) &P[0].Mass  - (char*) P,
        /* Regions allocated inside _prepare*/
        NULL,
        /* By default all particles are active. For hybrid neutrinos set below.*/
        NULL,
        PartManager->NumPart,
    };

    PetaPMGlobalFunctions global_functions = {NULL, NULL, potential_transfer};


    int i;
    #pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i++)
    {
        P[i].GravPM[0] = P[i].GravPM[1] = P[i].GravPM[2] = 0;
    }

    /* Set up parameters*/
    GravPM.Time = Time;
    GravPM.TimeIC = TimeIC;
    GravPM.CP = CP;
    GravPM.FastParticleType = FastParticleType;
    GravPM.UnitLength_in_cm = UnitLength_in_cm;
    /*
     * we apply potential transfer immediately after the R2C transform,
     * Therefore the force transfer functions are based on the potential,
     * not the density.
     * */
     
    gravpm_set_lbox_nonperiodic();  /* Find the New box size (GravPM.BoxSize is already double the particle range) */
    petapm_force(pm, _prepare, &global_functions, functions, &pstruct, tree);
    powerspectrum_sum(pm->ps);
    /*Now save the power spectrum*/
    powerspectrum_save(pm->ps, PowerOutputDir, "powerspectrum", Time, GrowthFactor(CP, Time, 1.0));
    /* Save the neutrino power if it is allocated*/
    if(pm->ps->logknu)
        powerspectrum_nu_save(pm->ps, PowerOutputDir, "powerspectrum-nu", Time);
    /*We are done with the power spectrum, free it*/
    powerspectrum_free(pm->ps);
    walltime_measure("/LongRange");
}

static PetaPMRegion * _prepare(PetaPM * pm, PetaPMParticleStruct * pstruct, void * userdata, int * Nregions) {
    /*
     *
     * walks down the tree, identify nodes that contains local mass and
     * are sufficiently large in volume.
     *
     * for each nodes, a mesh region is created.
     * the particles in a node are linked to their hosting region
     * (each particle belongs
     * to exactly one region even though it may be covered by two)
     *
     * */
    ForceTree * tree = (ForceTree *) userdata;
    /* In worst case, each topleave becomes a region: thus
     * NTopLeaves is sufficient */
    PetaPMRegion * regions = (PetaPMRegion *) mymalloc2("Regions", sizeof(PetaPMRegion) * tree->NTopLeaves);
    pstruct->RegionInd = (int *) mymalloc2("RegionInd", PartManager->NumPart * sizeof(int));
    int r = 0;

    int no = tree->firstnode; /* start with the root */
    while(no >= 0) {

        if(!(tree->Nodes[no].f.DependsOnLocalMass)) {
            /* node doesn't contain particles on this process, do not open */
            no = tree->Nodes[no].sibling;
            continue;
        }
        if(
            /* node is large */
           (tree->Nodes[no].len <= GravPM->BoxSize / pm->Nmesh * 24)
           ||
            /* node is a top leaf */
            ( !tree->Nodes[no].f.InternalTopLevel && (tree->Nodes[no].f.TopLevel) )
                ) {
            regions[r].no = no;
            r ++;
            /* do not open */
            no = tree->Nodes[no].sibling;
            continue;
        }
        /* open */
        no = tree->Nodes[no].s.suns[0];
    }

    *Nregions = r;
    int maxNregions;
    MPI_Reduce(&r, &maxNregions, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    message(0, "max number of regions is %d\n", maxNregions);

    int64_t i;
    int numswallowed = 0;
    #pragma omp parallel for reduction(+: numswallowed)
    for(i =0; i < PartManager->NumPart; i ++) {
        /* Swallowed black hole particles stick around but should not gravitate.
         * Short-range is handled by not adding them to the tree. */
        if(P[i].Swallowed){
            pstruct->RegionInd[i] = -2;
            numswallowed++;
        }
        else
            pstruct->RegionInd[i] = -1;
    }

    /* now lets mark particles to their hosting region */
    int64_t numpart = 0;
#pragma omp parallel for reduction(+: numpart)
    for(r = 0; r < *Nregions; r++) {
        regions[r].numpart = pm_mark_region_for_node(regions[r].no, r, pstruct->RegionInd, tree);
        numpart += regions[r].numpart;
    }
    for(i = 0; i < PartManager->NumPart; i ++) {
        if(pstruct->RegionInd[i] == -1) {
            message(1, "i = %d father %d not assigned to a region\n", i, force_get_father(i, tree));
        }
    }
    /* All particles shall have been processed just once. Otherwise we die */
    if((numpart+numswallowed) != PartManager->NumPart) {
        endrun(1, "Processed only %ld particles out of %ld\n", numpart, PartManager->NumPart);
    }
    for(r =0; r < *Nregions; r++) {
        convert_node_to_region(pm, &regions[r], tree->Nodes);
    }
    /*This is done to conserve memory during the PM step*/
    if(force_tree_allocated(tree)) force_tree_free(tree);

    /*Allocate memory for a power spectrum*/
    powerspectrum_alloc(pm->ps, pm->Nmesh, omp_get_max_threads(), GravPM.CP->MassiveNuLinRespOn, GravPM.BoxSize*GravPM.UnitLength_in_cm);

    walltime_measure("/PMgrav/Regions");
    return regions;
}

static int pm_mark_region_for_node(int startno, int rid, int * RegionInd, const ForceTree * tree) {
    int numpart = 0;
    int no = startno;
    int endno = tree->Nodes[startno].sibling;
    while(no >= 0 && no != endno)
    {
        struct NODE * nop = &tree->Nodes[no];
        if(nop->f.ChildType == PARTICLE_NODE_TYPE) {
            int i;
            for(i = 0; i < nop->s.noccupied; i++) {
                int p = nop->s.suns[i];
                RegionInd[p] = rid;
#ifdef DEBUG
                /* Check for particles outside of the node. This should never happen,
                * unless there is a bug in tree build, or the particles are being moved.*/
                int k;
                for(k = 0; k < 3; k ++) {
                    double l = P[p].Pos[k] - tree->Nodes[startno].center[k];
                    l = fabs(l * 2);
                    if (l > tree->Nodes[startno].len) {
                        if(l > tree->Nodes[startno].len * (1+ 1e-7))
                        endrun(1, "enlarging node size from %g to %g, due to particle of type %d at %g %g %g id=%ld\n",
                            tree->Nodes[startno].len, l, P[p].Type, P[p].Pos[0], P[p].Pos[1], P[p].Pos[2], P[p].ID);
                        tree->Nodes[startno].len = l;
                    }
                }
#endif
            }
            numpart += nop->s.noccupied;
            /* Move to sibling*/
            no = nop->sibling;
        }
        /* Generally this function should be handed top leaves which do not contain pseudo particles.
         * However, sometimes it will receive an internal top node which can, if that top node is small enough.
         * In this case, just move to the sibling: no particles to mark here.*/
        else if(nop->f.ChildType == PSEUDO_NODE_TYPE)
            no = nop->sibling;
        else if(nop->f.ChildType == NODE_NODE_TYPE)
            no = nop->s.suns[0];
        else
            endrun(122, "Unrecognised Node type %d, memory corruption!\n", nop->f.ChildType);
    }
    return numpart;
}


static void convert_node_to_region(PetaPM * pm, PetaPMRegion * r, struct NODE * Nodes) {
    int k;
    double cellsize = GravPM.BoxSize / pm->Nmesh;
    int no = r->no;
#if 0
    printf("task = %d no = %d len = %g hmax = %g center = %g %g %g\n",
            ThisTask, no, Nodes[no].len, Nodes[no].hmax,
            Nodes[no].center[0],
            Nodes[no].center[1],
            Nodes[no].center[2]);
#endif
    for(k = 0; k < 3; k ++) {
        r->offset[k] = floor((Nodes[no].center[k] - Nodes[no].len * 0.5) / cellsize);
        int end = (int) ceil((Nodes[no].center[k] + Nodes[no].len * 0.5) / cellsize) + 1;
        r->size[k] = end - r->offset[k] + 1;
        r->center[k] = Nodes[no].center[k];
    }

    /* setup the internal data structure of the region */
    petapm_region_init_strides(r);

    r->len  = Nodes[no].len;
}

/********************
 * transfer functions for
 *
 * potential from mass in cell
 *
 * and
 *
 * force from potential
 *
 *********************/

/* unnormalized sinc function sin(x) / x */
static double sinc_unnormed(double x) {
    if(x < 1e-5 && x > -1e-5) {
        double x2 = x * x;
        return 1.0 - x2 / 6. + x2  * x2 / 120.;
    } else {
        return sin(x) / x;
    }
}



/*Just read the power spectrum, without changing the input value.*/
void
measure_power_spectrum(PetaPM * pm, int64_t k2, int kpos[3], pfft_complex *value) {
    double f = 1.0;
    /* the CIC deconvolution kernel is
     *
     * sinc_unnormed(k_x L / 2 Nmesh) ** 2
     *
     * k_x = kpos * 2pi / L
     *
     * */
    int k;
    for(k = 0; k < 3; k ++) {
        double tmp = (kpos[k] * M_PI) / pm->Nmesh;
        tmp = sinc_unnormed(tmp);
        f *= 1. / (tmp * tmp);
    }
    powerspectrum_add_mode(pm->ps, k2, kpos, value, f, pm->Nmesh);
}

static void
potential_transfer(PetaPM * pm, int64_t k2, int kpos[3], pfft_complex *value)
{
    const double asmth2 = pow((2 * M_PI) * pm->Asmth / pm->Nmesh,2);
    double f = 1.0;
    const double smth = exp(-k2 * asmth2) / k2;
        /* fac is - 4pi G     (L / 2pi) **2 / L ** 3
     *        Gravity       k2            DFT (dk **3, but )
     * */
    const double pot_factor = - pm->G / (M_PI * GravPM.BoxSize);	/* to get potential */


    /* the CIC deconvolution kernel is
     *
     * sinc_unnormed(k_x L / 2 Nmesh) ** 2
     *
     * k_x = kpos * 2pi / L
     *
     * */
    int k;
    for(k = 0; k < 3; k ++) {
        double tmp = (kpos[k] * M_PI) / pm->Nmesh;
        tmp = sinc_unnormed(tmp);
        f *= 1. / (tmp * tmp);
    }
    /*
     * first decovolution is CIC in par->mesh
     * second decovolution is correcting readout
     * I don't understand the second yet!
     * */
    const double fac = pot_factor * smth * f * f;
    Power * ps = pm->ps;

    /*Add neutrino power if desired*/
    if(GravPM.CP->MassiveNuLinRespOn && k2 > 0) {
        /* Change the units of k to match those of logkk*/
        double logk2 = log(sqrt(k2) * 2 * M_PI / ps->BoxSize_in_MPC);
        /* Floating point roundoff and the binning means there may be a mode just beyond the box size.*/
        if(logk2 < ps->logknu[0] && logk2 > ps->logknu[0]-log(2) )
            logk2 = ps->logknu[0];
        else if( logk2 > ps->logknu[ps->nonzero-1])
            logk2 = ps->logknu[ps->nonzero-1];
        /* Note get_neutrino_powerspec returns Omega_nu / (Omega0 -OmegaNu) * delta_nu / P_cdm^1/2, which is dimensionless.
         * So below is: M_cdm * delta_cdm (1 + Omega_nu/(Omega0-OmegaNu) (delta_nu / delta_cdm))
         *            = M_cdm * (delta_cdm (Omega0 - OmegaNu)/Omega0 + Omega_nu/Omega0 delta_nu) * Omega0 / (Omega0-OmegaNu)
         *            = M_cdm * Omega0 / (Omega0-OmegaNu) * (delta_cdm (1 - f_nu)  + f_nu delta_nu) )
         *            = M_cdm * Omega0 / (Omega0-OmegaNu) * delta_t
         *            = (M_cdm + M_nu) * delta_t
         * This is correct for the forces, and gives the right power spectrum,
         * once we multiply PowerSpectrum.Norm by (Omega0 / (Omega0 - OmegaNu))**2 */
        const double nufac = 1 + ps->nu_prefac * gsl_interp_eval(ps->nu_spline,ps->logknu,
                                                                       ps->delta_nu_ratio,logk2,ps->nu_acc);
        value[0][0] *= nufac;
        value[0][1] *= nufac;
    }

    /*Compute the power spectrum*/
    powerspectrum_add_mode(ps, k2, kpos, value, f, pm->Nmesh);
    if(k2 == 0) {
        if(GravPM.CP->MassiveNuLinRespOn) {
            const double MtotbyMcdm = GravPM.CP->Omega0/(GravPM.CP->Omega0 - pow(GravPM.Time,3)*get_omega_nu_nopart(&(GravPM.CP->ONu), GravPM.Time));
            ps->Norm *= MtotbyMcdm*MtotbyMcdm;
        }
        /* Remove zero mode corresponding to the mean.*/
        value[0][0] = 0.0;
        value[0][1] = 0.0;
        return;
    }

    value[0][0] *= fac;
    value[0][1] *= fac;
}

/* the transfer functions for force in fourier space applied to potential */
/* super lanzcos in CH6 P 122 Digital Filters by Richard W. Hamming */
static double diff_kernel(double w) {
/* order N = 1 */
/*
 * This is the same as GADGET-2 but in fourier space:
 * see gadget-2 paper and Hamming's book.
 * c1 = 2 / 3, c2 = 1 / 12
 * */
    return 1 / 6.0 * (8 * sin (w) - sin (2 * w));
}

/*This function decides if a particle is actively gravitating; tracers are not.*/
static int hybrid_nu_gravpm_is_active(int i) {
    if (P[i].Type == GravPM.FastParticleType)
        return 0;
    else
        return 1;
}

static void force_transfer(PetaPM * pm, int k, pfft_complex * value) {
    double tmp0;
    double tmp1;
    /*
     * negative sign is from force_x = - Del_x pot
     *
     * filter is   i K(w)
     * */
    double fac = -1 * diff_kernel (k * (2 * M_PI / pm->Nmesh)) * (pm->Nmesh / GravPM.BoxSize);
    tmp0 = - value[0][1] * fac;
    tmp1 = value[0][0] * fac;
    value[0][0] = tmp0;
    value[0][1] = tmp1;
}
static void force_x_transfer(PetaPM * pm, int64_t k2, int kpos[3], pfft_complex * value) {
    force_transfer(pm, kpos[0], value);
}
static void force_y_transfer(PetaPM * pm, int64_t k2, int kpos[3], pfft_complex * value) {
    force_transfer(pm, kpos[1], value);
}
static void force_z_transfer(PetaPM * pm, int64_t k2, int kpos[3], pfft_complex * value) {
    force_transfer(pm, kpos[2], value);
}
static void readout_potential(PetaPM * pm, int i, double * mesh, double weight) {
    P[i].Potential += weight * mesh[0];
}
static void readout_force_x(PetaPM * pm, int i, double * mesh, double weight) {
    P[i].GravPM[0] += weight * mesh[0];
}
static void readout_force_y(PetaPM * pm, int i, double * mesh, double weight) {
    P[i].GravPM[1] += weight * mesh[0];
}
static void readout_force_z(PetaPM * pm, int i, double * mesh, double weight) {
    P[i].GravPM[2] += weight * mesh[0];
}
