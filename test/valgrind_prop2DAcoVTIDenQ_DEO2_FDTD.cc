#include "stdio.h"
#include "prop2DAcoVTIDenQ_DEO2_FDTD.h"

void test() {
    long freeSurface = 1;
    long nthread = 1;
    long nx = 51; 
    long nz = 41;
    long nbx = 51; 
    long nbz = 8;
    long nsponge = 10;
    float dx = 25;
    float dz = 25;
    float dt = 0.001;

    Prop2DAcoVTIDenQ_DEO2_FDTD *op = new Prop2DAcoVTIDenQ_DEO2_FDTD(freeSurface, 
        nthread, nx, nz, nsponge, dx, dz, dt, nbx, nbz);

    float *dVel = new float[nx * nz];
    float *dEps = new float[nx * nz];
    float *dEta = new float[nx * nz];
    float *wavefieldP = new float[nx * nz];
    float *wavefieldM = new float[nx * nz];
    float *wavefieldDP = new float[nx * nz];
    float *wavefieldDM = new float[nx * nz];
    float *inPX = new float[nx * nz];
    float *inPZ = new float[nx * nz];
    float *inMX = new float[nx * nz];
    float *inMZ = new float[nx * nz];
    float *fieldVel = new float[nx * nz];
    float *fieldEps = new float[nx * nz];
    float *fieldEta = new float[nx * nz];
    float *fieldVsVp = new float[nx * nz];
    float *fieldBuoy = new float[nx * nz];
    float *outPX = new float[nx * nz];
    float *outPZ = new float[nx * nz];
    float *outMX = new float[nx * nz];
    float *outMZ = new float[nx * nz];
    float *dtOmegaInvQ = new float[nx * nz];
    float *pCur = new float[nx * nz];
    float *pSpace = new float[nx * nz];
    float *pOld = new float[nx * nz];
    float *mCur = new float[nx * nz];
    float *mSpace = new float[nx * nz];
    float *mOld = new float[nx * nz];

    for (long k = 0; k < nx * nz; k++) {
        dVel[k] = 1;
        dEps[k] = 1;
        dEta[k] = 1;
        wavefieldP[k] = 1;
        wavefieldM[k] = 1;
        wavefieldDP[k] = 1;
        wavefieldDM[k] = 1;
        inPX[k] = 1;
        inPZ[k] = 1;
        inMX[k] = 1;
        inMZ[k] = 1;
        fieldVel[k] = 1;
        fieldEps[k] = 1;
        fieldEta[k] = 1;
        fieldVsVp[k] = 1;
        fieldBuoy[k] = 1;
        outPX[k] = 1;
        outPZ[k] = 1;
        outMX[k] = 1;
        outMZ[k] = 1;
        dtOmegaInvQ[k] = 1;
        pCur[k] = 1;
        pSpace[k] = 1;
        pOld[k] = 1;
        mCur[k] = 1;
        mSpace[k] = 1;
        mOld[k] = 1;
    }

    op->timeStep();
    op->timeStepLinear();
    op->scaleSpatialDerivatives();
    op->forwardBornInjection_V(dVel, wavefieldDP, wavefieldDM);
    op->forwardBornInjection_VEA(dVel, dEps, dEta, wavefieldP, wavefieldM, wavefieldDP, wavefieldDM);
    op->adjointBornAccumulation_V(dVel, wavefieldDP, wavefieldDM);
    op->adjointBornAccumulation_wavefieldsep_V(dVel, wavefieldDP, wavefieldDM, 0);
    op->adjointBornAccumulation_wavefieldsep_V(dVel, wavefieldDP, wavefieldDM, 1);
    op->adjointBornAccumulation_VEA(dVel, dEps, dEta, wavefieldP, wavefieldM, wavefieldDP, wavefieldDM);
    op->applyFirstDerivatives2D_MinusHalf_TimeUpdate_Nonlinear(freeSurface, nx, nz, nthread,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, outPX, outPZ, outMX, outMZ, fieldVel, fieldBuoy, dtOmegaInvQ, pCur, mCur, pSpace, mSpace, pOld, mOld, nbx, nbz);
    op->applyFirstDerivatives2D_MinusHalf_TimeUpdate_Linear(freeSurface, nx, nz, nthread,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, outPX, outPZ, outMX, outMZ, fieldVel, fieldBuoy, dtOmegaInvQ, pCur, mCur, pOld, mOld, nbx, nbz);
    op->applyFirstDerivatives2D_PlusHalf(freeSurface, nx, nz, nthread,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, inPX, inPZ, outPX, outPZ, nbx, nbz);
    op->applyFirstDerivatives2D_MinusHalf(freeSurface, nx, nz, nthread,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, inPX, inPZ, outPX, outPZ, nbx, nbz);

    delete [] dVel;
    delete [] dEps;
    delete [] dEta;
    delete [] wavefieldP;
    delete [] wavefieldM;
    delete [] wavefieldDP;
    delete [] wavefieldDM;
    delete [] inPX;
    delete [] inPZ;
    delete [] inMX;
    delete [] inMZ;
    delete [] fieldVel;
    delete [] fieldEps;
    delete [] fieldEta;
    delete [] fieldVsVp;
    delete [] fieldBuoy;
    delete [] outPX;
    delete [] outPZ;
    delete [] outMX;
    delete [] outMZ;
    delete [] dtOmegaInvQ;
    delete [] pCur;
    delete [] pSpace;
    delete [] pOld;
    delete [] mCur;
    delete [] mSpace;
    delete [] mOld;

    delete op;
}

int main(int argc, char **argv) {
    test();
}