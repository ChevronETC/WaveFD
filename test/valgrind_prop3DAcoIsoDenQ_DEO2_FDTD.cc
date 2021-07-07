#include "stdio.h"
#include "prop3DAcoIsoDenQ_DEO2_FDTD.h"

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

    Prop3DAcoIsoDenQ_DEO2_FDTD *op = new Prop3DAcoIsoDenQ_DEO2_FDTD(freeSurface, 
        nthread, nx, ny, nz, nsponge, dx, dy, dz, dt, nbx, nby, nbz);

    float *dVel = new float[nx * ny * nz];
    float *wavefieldDP = new float[nx * ny * nz];
    float *inPX = new float[nx * ny * nz];
    float *inPY = new float[nx * ny * nz];
    float *inPZ = new float[nx * ny * nz];
    float *fieldVel = new float[nx * ny * nz];
    float *fieldBuoy = new float[nx * ny * nz];
    float *tmpPX = new float[nx * ny * nz];
    float *tmpPY = new float[nx * ny * nz];
    float *tmpPZ = new float[nx * ny * nz];
    float *dtOmegaInvQ = new float[nx * ny * nz];
    float *pCur = new float[nx * ny * nz];
    float *pSpace = new float[nx * ny * nz];
    float *pOld = new float[nx * ny * nz];

    for (long k = 0; k < nx * ny * nz; k++) {
        dVel[k] = 1;
        wavefieldDP[k] = 1;
        inPX[k] = 1;
        inPY[k] = 1;
        inPZ[k] = 1;
        fieldVel[k] = 1;
        fieldBuoy[k] = 1;
        tmpPX[k] = 1;
        tmpPY[k] = 1;
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
    op->applyFirstDerivatives3D_PlusHalf_Sandwich_Isotropic(freeSurface, nx, ny, nz, nthread, 
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, inPX, inPY, inPZ, fieldBuoy, tmpPX, tmpPY, tmpPZ, nbx, nby, nbz);
    op->applyFirstDerivatives3D_MinusHalf_TimeUpdate_Nonlinear_Isotropic(freeSurface, nx, ny, nz, nthread, 
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        tmpPX, tmpPY, tmpPZ, fieldVel, fieldBuoy, dtOmegaInvQ, pCur, pSpace, pOld, nbx, nby, nbz);
    op->scaleSpatialDerivatives();

    delete [] dVel;
    delete [] wavefieldDP;
    delete [] inPX;
    delete [] inPY;
    delete [] inPZ;
    delete [] fieldVel;
    delete [] fieldBuoy;
    delete [] tmpPX;
    delete [] tmpPY;
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