#include "stdio.h"
#include "prop3DAcoTTIDenQ_DEO2_FDTD.h"

void test() {
    long freeSurface = 1;
    long nthread = 1;
    long nx = 51; 
    long ny = 51; 
    long nz = 51;
    long nbx = 51; 
    long nby = 8; 
    long nbz = 8;
    long nsponge = 10;
    float dx = 25;
    float dy = 25;
    float dz = 25;
    float dt = 0.001;

    Prop3DAcoTTIDenQ_DEO2_FDTD *op = new Prop3DAcoTTIDenQ_DEO2_FDTD(freeSurface, 
        nthread, nx, ny, nz, nsponge, dx, dy, dz, dt, nbx, nby, nbz);

    float *dVel = new float[nx * ny * nz];
    float *dEps = new float[nx * ny * nz];
    float *dEta = new float[nx * ny * nz];
    float *wavefieldP = new float[nx * ny * nz];
    float *wavefieldM = new float[nx * ny * nz];
    float *wavefieldDP = new float[nx * ny * nz];
    float *wavefieldDM = new float[nx * ny * nz];
    float *inPG1 = new float[nx * ny * nz];
    float *inPG2 = new float[nx * ny * nz];
    float *inPG3 = new float[nx * ny * nz];
    float *inMG1 = new float[nx * ny * nz];
    float *inMG2 = new float[nx * ny * nz];
    float *inMG3 = new float[nx * ny * nz];
    float *sinTheta = new float[nx * ny * nz];
    float *cosTheta = new float[nx * ny * nz];
    float *sinPhi = new float[nx * ny * nz];
    float *cosPhi = new float[nx * ny * nz];
    float *fieldVel = new float[nx * ny * nz];
    float *fieldEps = new float[nx * ny * nz];
    float *fieldEta = new float[nx * ny * nz];
    float *fieldVsVp = new float[nx * ny * nz];
    float *fieldBuoy = new float[nx * ny * nz];
    float *outPG1 = new float[nx * ny * nz];
    float *outPG2 = new float[nx * ny * nz];
    float *outPG3 = new float[nx * ny * nz];
    float *outMG1 = new float[nx * ny * nz];
    float *outMG2 = new float[nx * ny * nz];
    float *outMG3 = new float[nx * ny * nz];
    float *dtOmegaInvQ = new float[nx * ny * nz];
    float *pCur = new float[nx * ny * nz];
    float *pSpace = new float[nx * ny * nz];
    float *pOld = new float[nx * ny * nz];
    float *mCur = new float[nx * ny * nz];
    float *mSpace = new float[nx * ny * nz];
    float *mOld = new float[nx * ny * nz];

    for (long k = 0; k < nx * ny * nz; k++) {
        dVel[k] = 1;
        dEps[k] = 1;
        dEta[k] = 1;
        wavefieldP[k] = 1;
        wavefieldM[k] = 1;
        wavefieldDP[k] = 1;
        wavefieldDM[k] = 1;
        inPG1[k] = 1;
        inPG2[k] = 1;
        inPG3[k] = 1;
        inMG1[k] = 1;
        inMG2[k] = 1;
        inMG3[k] = 1;
        sinTheta[k] = 1;
        cosTheta[k] = 1;
        sinPhi[k] = 1;
        cosPhi[k] = 1;
        fieldVel[k] = 1;
        fieldEps[k] = 1;
        fieldEta[k] = 1;
        fieldVsVp[k] = 1;
        fieldBuoy[k] = 1;
        outPG1[k] = 1;
        outPG2[k] = 1;
        outPG3[k] = 1;
        outMG1[k] = 1;
        outMG2[k] = 1;
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
    op->applyRotationSandwichRotation_TTI_FirstDerivatives3D_PlusHalf_TwoFields(freeSurface, nx, ny, nz, nthread,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, inPG1, inMG1, sinTheta, cosTheta, sinPhi, cosPhi, fieldEps, fieldEta, fieldVsVp, fieldBuoy, outPG1, outPG2, outPG3, outMG1, outMG2, outMG3, nbx, nby, nbz);
    op->applyFirstDerivatives3D_MinusHalf_TimeUpdate_Nonlinear(freeSurface, nx, ny, nz, nthread,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, inPG1, inPG2, inPG3, inMG1, inMG2, inMG3, fieldVel, fieldBuoy, dtOmegaInvQ, pCur, mCur, pSpace, mSpace, pOld, mOld, nbx, nby, nbz);
    op->applyFirstDerivatives3D_TTI_PlusHalf_Sandwich(freeSurface, nx, ny, nz, nthread,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, inPG1, inPG2, inPG3, inMG1, inMG2, inMG3, fieldEps, fieldEta, fieldVsVp, fieldBuoy, sinTheta, cosTheta, sinPhi, cosPhi, outPG1, outPG2, outPG3, outMG1, outMG2, outMG3, nbx, nby, nbz);
    op->applyFirstDerivatives3D_TTI_MinusHalf_TimeUpdate_Nonlinear(freeSurface, nx, ny, nz, nthread,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, dt, inPG1, inPG2, inPG3, inMG1, inMG2, inMG3, fieldVel, fieldBuoy, dtOmegaInvQ, sinTheta, cosTheta, sinPhi, cosPhi, pCur, mCur, pSpace, mSpace, pOld, mOld, nbx, nby, nbz);
    op->applyFirstDerivatives3D_TTI_PlusHalf(freeSurface, nx, ny, nz, nthread,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, inPG1, inPG2, inPG3, sinTheta, cosTheta, sinPhi, cosPhi, outPG1, outPG2, outPG3, nbx, nby, nbz);
    op->applyFirstDerivatives3D_TTI_MinusHalf(freeSurface, nx, ny, nz, nthread,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, inPG1, inPG2, inPG3, sinTheta, cosTheta, sinPhi, cosPhi, outPG1, outPG2, outPG3, nbx, nby, nbz);

    delete [] dVel;
    delete [] dEps;
    delete [] dEta;
    delete [] wavefieldP;
    delete [] wavefieldM;
    delete [] wavefieldDP;
    delete [] wavefieldDM;
    delete [] inPG1;
    delete [] inPG2;
    delete [] inPG3;
    delete [] inMG1;
    delete [] inMG2;
    delete [] inMG3;
    delete [] sinTheta;
    delete [] cosTheta;
    delete [] sinPhi;
    delete [] cosPhi;
    delete [] fieldVel;
    delete [] fieldEps;
    delete [] fieldEta;
    delete [] fieldVsVp;
    delete [] fieldBuoy;
    delete [] outPG1;
    delete [] outPG2;
    delete [] outPG3;
    delete [] outMG1;
    delete [] outMG2;
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