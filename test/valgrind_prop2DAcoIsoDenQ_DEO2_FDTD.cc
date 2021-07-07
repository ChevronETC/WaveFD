#include "stdio.h"
#include "prop2DAcoIsoDenQ_DEO2_FDTD.h"

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

    Prop2DAcoIsoDenQ_DEO2_FDTD *op = new Prop2DAcoIsoDenQ_DEO2_FDTD(freeSurface, 
        nthread, nx, nz, nsponge, dx, dz, dt, nbx, nbz);

    float *dVel = new float[nx * nz];
    float *wavefieldDP = new float[nx * nz];
    float *inPX = new float[nx * nz];
    float *inPZ = new float[nx * nz];
    float *fieldVel = new float[nx * nz];
    float *fieldBuoy = new float[nx * nz];
    float *tmpPX = new float[nx * nz];
    float *tmpPZ = new float[nx * nz];
    float *dtOmegaInvQ = new float[nx * nz];
    float *pCur = new float[nx * nz];
    float *pSpace = new float[nx * nz];
    float *pOld = new float[nx * nz];

    for (long k = 0; k < nx * nz; k++) {
        dVel[k] = 1;
        wavefieldDP[k] = 1;
        inPX[k] = 1;
        inPZ[k] = 1;
        fieldVel[k] = 1;
        fieldBuoy[k] = 1;
        tmpPX[k] = 1;
        tmpPZ[k] = 1;
        dtOmegaInvQ[k] = 1;
        pCur[k] = 1;
        pSpace[k] = 1;
        pOld[k] = 1;
    }

    op->timeStep();
    op->forwardBornInjection(dVel, wavefieldDP);
    op->adjointBornAccumulation(dVel, wavefieldDP);
    op->adjointBornAccumulation_wavefieldsep(dVel, wavefieldDP, 0);
    op->adjointBornAccumulation_wavefieldsep(dVel, wavefieldDP, 1);
    op->applyFirstDerivatives2D_PlusHalf_Sandwich(freeSurface, nx, nz, nthread, 
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, inPX, inPZ, fieldBuoy, tmpPX, tmpPZ, nbx, nbz);
    op->applyFirstDerivatives2D_MinusHalf_TimeUpdate_Nonlinear(freeSurface, nx, nz, nthread, 
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        tmpPX, tmpPZ, fieldVel, fieldBuoy, dtOmegaInvQ, pCur, pSpace, pOld, nbx, nbz);
    op->scaleSpatialDerivatives();

    delete [] dVel;
    delete [] wavefieldDP;
    delete [] inPX;
    delete [] inPZ;
    delete [] fieldVel;
    delete [] fieldBuoy;
    delete [] tmpPX;
    delete [] tmpPZ;
    delete [] dtOmegaInvQ;
    delete [] pCur;
    delete [] pSpace;
    delete [] pOld;
    
    delete op;
}

int main(int argc, char **argv) {
    test();
}