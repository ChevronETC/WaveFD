#include "stdio.h"
#include "prop2DAcoTTIDenQ_DEO2_FDTD.h"

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

    Prop2DAcoTTIDenQ_DEO2_FDTD *op = new Prop2DAcoTTIDenQ_DEO2_FDTD(freeSurface, 
        nthread, nx, nz, nsponge, dx, dz, dt, nbx, nbz);

    float *dVel = new float[nx * nz];
    float *dEps = new float[nx * nz];
    float *dEta = new float[nx * nz];
    float *wavefieldP = new float[nx * nz];
    float *wavefieldM = new float[nx * nz];
    float *wavefieldDP = new float[nx * nz];
    float *wavefieldDM = new float[nx * nz];
    float *inPG1 = new float[nx * nz];
    float *inPG3 = new float[nx * nz];
    float *inMG1 = new float[nx * nz];
    float *inMG3 = new float[nx * nz];
    float *sinTheta = new float[nx * nz];
    float *cosTheta = new float[nx * nz];
    float *fieldVel = new float[nx * nz];
    float *fieldEps = new float[nx * nz];
    float *fieldEta = new float[nx * nz];
    float *fieldVsVp = new float[nx * nz];
    float *fieldBuoy = new float[nx * nz];
    float *outPG1 = new float[nx * nz];
    float *outPG3 = new float[nx * nz];
    float *outMG1 = new float[nx * nz];
    float *outMG3 = new float[nx * nz];
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
        inPG1[k] = 1;
        inPG3[k] = 1;
        inMG1[k] = 1;
        inMG3[k] = 1;
        sinTheta[k] = 1;
        cosTheta[k] = 1;
        fieldVel[k] = 1;
        fieldEps[k] = 1;
        fieldEta[k] = 1;
        fieldVsVp[k] = 1;
        fieldBuoy[k] = 1;
        outPG1[k] = 1;
        outPG3[k] = 1;
        outMG1[k] = 1;
        outMG3[k] = 1;
        dtOmegaInvQ[k] = 1;
        pCur[k] = 1;
        pSpace[k] = 1;
        pOld[k] = 1;
        mCur[k] = 1;
        mSpace[k] = 1;
        mOld[k] = 1;
    }

    op->timeStep();
    op->scaleSpatialDerivatives();
    op->forwardBornInjection_V(dVel, wavefieldDP, wavefieldDM);
    op->forwardBornInjection_VEA(dVel, dEps, dEta, wavefieldP, wavefieldM, wavefieldDP, wavefieldDM);
    op->adjointBornAccumulation_V(dVel, wavefieldDP, wavefieldDM);
    op->adjointBornAccumulation_wavefieldsep_V(dVel, wavefieldDP, wavefieldDM, 0);
    op->adjointBornAccumulation_wavefieldsep_V(dVel, wavefieldDP, wavefieldDM, 1);
    op->adjointBornAccumulation_VEA(dVel, dEps, dEta, wavefieldP, wavefieldM, wavefieldDP, wavefieldDM);
    op->applyFirstDerivatives2D_TTI_PlusHalf_Sandwich(freeSurface, nx, nz, nthread,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, inPG1, inPG3, inMG1, inMG3, sinTheta, cosTheta, fieldEps, fieldEta, fieldVsVp, fieldBuoy, outPG1, outPG3, outMG1, outMG3, nbx, nbz);
    op->applyFirstDerivatives2D_TTI_MinusHalf_TimeUpdate_Nonlinear(freeSurface, nx, nz, nthread,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, inPG1, inPG3, inMG1, inMG3, sinTheta, cosTheta, fieldVel, fieldBuoy, dtOmegaInvQ, pCur, mCur, pSpace, mSpace, pOld, mOld, nbx, nbz);
    op->applyFirstDerivatives2D_TTI_PlusHalf(freeSurface, nx, nz, nthread,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, inPG1, inPG3, sinTheta, cosTheta, outPG1, outPG3, nbx, nbz);
    op->applyFirstDerivatives2D_TTI_MinusHalf(freeSurface, nx, nz, nthread,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, inPG1, inPG3, sinTheta, cosTheta, outPG1, outPG3, nbx, nbz);

    delete [] dVel;
    delete [] dEps;
    delete [] dEta;
    delete [] wavefieldP;
    delete [] wavefieldM;
    delete [] wavefieldDP;
    delete [] wavefieldDM;
    delete [] inPG1;
    delete [] inPG3;
    delete [] inMG1;
    delete [] inMG3;
    delete [] sinTheta;
    delete [] cosTheta;
    delete [] fieldVel;
    delete [] fieldEps;
    delete [] fieldEta;
    delete [] fieldVsVp;
    delete [] fieldBuoy;
    delete [] outPG1;
    delete [] outPG3;
    delete [] outMG1;
    delete [] outMG3;
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