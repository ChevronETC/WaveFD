#ifndef PROP3DACOISODENQ_DEO2_FDTD_H
#define PROP3DACOISODENQ_DEO2_FDTD_H

#include <omp.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <fftw3.h>
#include <complex>

#include "propagatorStaticFunctions.h"

#define MIN(x,y) ((x)<(y)?(x):(y))

class Prop3DAcoIsoDenQ_DEO2_FDTD {

public:
    const bool _freeSurface;
    const long _nbx, _nby, _nbz, _nthread, _nx, _ny, _nz, _nsponge;
    const float _dx, _dy, _dz, _dt;
    const float _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDy, _invDz;

    float * __restrict__ _v = NULL;
    float * __restrict__ _b = NULL;
    float * __restrict__ _dtOmegaInvQ = NULL;
    float * __restrict__ _pSpace = NULL;
    float * __restrict__ _tmpPx1 = NULL;
    float * __restrict__ _tmpPy1 = NULL;
    float * __restrict__ _tmpPz1 = NULL;
    float * __restrict__ _tmpPx2 = NULL;
    float * __restrict__ _tmpPy2 = NULL;
    float * __restrict__ _tmpPz2 = NULL;
    float * _pOld = NULL;
    float * _pCur = NULL;

    Prop3DAcoIsoDenQ_DEO2_FDTD(
        bool freeSurface,
        long nthread,
        long nx,
        long ny,
        long nz,
        long nsponge,
        float dx,
        float dy,
        float dz,
        float dt,
        const long nbx,
        const long nby,
        const long nbz) :
            _freeSurface(freeSurface),
            _nthread(nthread),
            _nx(nx),
            _ny(ny),
            _nz(nz),
            _nsponge(nsponge),
            _nbx(nbx),
            _nby(nby),
            _nbz(nbz),
            _dx(dx),
            _dy(dy),
            _dz(dz),
            _dt(dt),
            _c8_1(+1225.0 / 1024.0),
            _c8_2(-245.0 / 3072.0),
            _c8_3(+49.0 / 5120.0),
            _c8_4(-5.0 / 7168.0),
            _invDx(1.0 / _dx),
            _invDy(1.0 / _dy),
            _invDz(1.0 / _dz) {

        // Allocate arrays
        _v           = new float[_nx * _ny * _nz];
        _b           = new float[_nx * _ny * _nz];
        _dtOmegaInvQ = new float[_nx * _ny * _nz];
        _pSpace      = new float[_nx * _ny * _nz];
        _tmpPx1      = new float[_nx * _ny * _nz];
        _tmpPy1      = new float[_nx * _ny * _nz];
        _tmpPz1      = new float[_nx * _ny * _nz];
        _tmpPx2      = new float[_nx * _ny * _nz];
        _tmpPy2      = new float[_nx * _ny * _nz];
        _tmpPz2      = new float[_nx * _ny * _nz];
        _pOld        = new float[_nx * _ny * _nz];
        _pCur        = new float[_nx * _ny * _nz];

        numaFirstTouch(_nx, _ny, _nz, _nthread, _v, _b,
            _dtOmegaInvQ, _pSpace, _tmpPx1, _tmpPy1, _tmpPz1, _tmpPx2, _tmpPy2, _tmpPz2, 
            _pOld, _pCur, _nbx, _nby, _nbz);
    }

#if defined(__FUNCTION_CLONES__)
__attribute__((target_clones("avx","avx2","avx512f","default")))
#endif
    inline void numaFirstTouch(
            const long nx,
            const long ny,
            const long nz,
            const long nthread,
            float * __restrict__ v,
            float * __restrict__ b,
            float * __restrict__ dtOmegaInvQ,
            float * __restrict__ pSpace,
            float * __restrict__ tmpPx1,
            float * __restrict__ tmpPy1,
            float * __restrict__ tmpPz1,
            float * __restrict__ tmpPx2,
            float * __restrict__ tmpPy2,
            float * __restrict__ tmpPz2,
            float * __restrict__ pOld,
            float * __restrict__ pCur,
            const long BX_3D,
            const long BY_3D,
            const long BZ_3D) {

        const long nx4 = nx - 4;
        const long ny4 = ny - 4;
        const long nz4 = nz - 4;

#pragma omp parallel for collapse(3) num_threads(nthread) schedule(static)
        for (long bx = 4; bx < nx4; bx += BX_3D) {
            for (long by = 4; by < ny4; by += BY_3D) {
                for (long bz = 4; bz < nz4; bz += BZ_3D) {
                    const long kxmax = MIN(bx + BX_3D, nx4);
                    const long kymax = MIN(by + BY_3D, ny4);
                    const long kzmax = MIN(bz + BZ_3D, nz4);

                    for (long kx = bx; kx < kxmax; kx++) {
                        for (long ky = by; ky < kymax; ky++) {
#pragma omp simd
                            for (long kz = bz; kz < kzmax; kz++) {
                                const long k = kx * _ny * _nz + ky * _nz + kz;

                                v[k] = 0;
                                b[k] = 0;
                                dtOmegaInvQ[k] = 0;
                                pSpace[k] = 0;
                                tmpPx1[k] = 0;
                                tmpPy1[k] = 0;
                                tmpPz1[k] = 0;
                                tmpPx2[k] = 0;
                                tmpPy2[k] = 0;
                                tmpPz2[k] = 0;
                                pOld[k] = 0;
                                pCur[k] = 0;
                            }
                        }
                    }
                }
            }
        }

        // annulus
        for (long k = 0; k < 4; k++) {
#pragma omp parallel for num_threads(nthread) schedule(static)
            for (long kx = 0; kx < nx; kx++) {
#pragma omp simd
                for (long ky = 0; ky < ny; ky++) {
                    const long kindex1 = kx * ny * nz + ky * nz + k;
                    const long kindex2 = kx * ny * nz + ky * nz + (nz - 1 - k);
                    v[kindex1] = b[kindex1] = dtOmegaInvQ[kindex1] = pSpace[kindex1] =
                        tmpPx1[kindex1] = tmpPy1[kindex1] = tmpPz1[kindex1] =
                        tmpPx2[kindex1] = tmpPy2[kindex1] = tmpPz2[kindex1] =
                        pOld[kindex1] = pCur[kindex1] = 0;
                    v[kindex2] = b[kindex2] = dtOmegaInvQ[kindex2] = pSpace[kindex2] =
                        tmpPx1[kindex2] = tmpPy1[kindex2] = tmpPz1[kindex2] =
                        tmpPx2[kindex2] = tmpPy2[kindex2] = tmpPz2[kindex2] =
                        pOld[kindex2] = pCur[kindex2] = 0;
                }
            }

#pragma omp parallel for num_threads(nthread) schedule(static)
            for (long kx = 0; kx < nx; kx++) {
#pragma omp simd
                for (long kz = 0; kz < nz; kz++) {
                    const long kindex1 = kx * ny * nz + k * nz + kz;
                    const long kindex2 = kx * ny * nz + (ny - 1 - k) * nz + kz;
                    v[kindex1] = b[kindex1] = dtOmegaInvQ[kindex1] = pSpace[kindex1] =
                        tmpPx1[kindex1] = tmpPy1[kindex1] = tmpPz1[kindex1] =
                        tmpPx2[kindex1] = tmpPy2[kindex1] = tmpPz2[kindex1] =
                        pOld[kindex1] = pCur[kindex1] = 0;
                    v[kindex2] = b[kindex2] = dtOmegaInvQ[kindex2] = pSpace[kindex2] =
                        tmpPx1[kindex2] = tmpPy1[kindex2] = tmpPz1[kindex2] =
                        tmpPx2[kindex2] = tmpPy2[kindex2] = tmpPz2[kindex2] =
                        pOld[kindex2] = pCur[kindex2] = 0;
                }
            }

#pragma omp parallel for num_threads(nthread) schedule(static)
            for (long ky = 0; ky < ny; ky++) {
#pragma omp simd
                for (long kz = 0; kz < nz; kz++) {
                    const long kindex1 = k * ny * nz + ky * nz + kz;
                    const long kindex2 = (nx - 1 - k) * ny * nz + ky * nz + kz;
                    v[kindex1] = b[kindex1] = dtOmegaInvQ[kindex1] = pSpace[kindex1] =
                        tmpPx1[kindex1] = tmpPy1[kindex1] = tmpPz1[kindex1] =
                        tmpPx2[kindex1] = tmpPy2[kindex1] = tmpPz2[kindex1] =
                        pOld[kindex1] = pCur[kindex1] = 0;
                    v[kindex2] = b[kindex2] = dtOmegaInvQ[kindex2] = pSpace[kindex2] =
                        tmpPx1[kindex2] = tmpPy1[kindex2] = tmpPz1[kindex2] =
                        tmpPx2[kindex2] = tmpPy2[kindex2] = tmpPz2[kindex2] =
                        pOld[kindex2] = pCur[kindex2] = 0;
                }
            }
        }
    }

    ~Prop3DAcoIsoDenQ_DEO2_FDTD() {
        if (_v != NULL) delete [] _v;
        if (_b != NULL) delete [] _b;
        if (_dtOmegaInvQ != NULL) delete [] _dtOmegaInvQ;
        if (_pSpace != NULL) delete [] _pSpace;
        if (_tmpPx1 != NULL) delete [] _tmpPx1;
        if (_tmpPy1 != NULL) delete [] _tmpPy1;
        if (_tmpPz1 != NULL) delete [] _tmpPz1;
        if (_tmpPx2 != NULL) delete [] _tmpPx2;
        if (_tmpPy2 != NULL) delete [] _tmpPy2;
        if (_tmpPz2 != NULL) delete [] _tmpPz2;
        if (_pOld != NULL) delete [] _pOld;
        if (_pCur != NULL) delete [] _pCur;
    }

#if defined(__FUNCTION_CLONES__)
__attribute__((target_clones("avx","avx2","avx512f","default")))
#endif
    void info() {
        printf("\n");
        printf("Prop3DAcoIsoDenQ_DEO2_FDTD\n");
        printf("  nx,ny,nz;           %5ld %5ld %5ld\n", _nx, _ny, _nz);
        printf("  nthread,nsponge,fs; %5ld %5ld %5d\n", _nthread, _nsponge, _freeSurface);
        printf("  X min,max,inc;    %+16.8f %+16.8f %+16.8f\n", 0.0, _dx * (_nx - 1), _dx);
        printf("  Y min,max,inc;    %+16.8f %+16.8f %+16.8f\n", 0.0, _dy * (_ny - 1), _dy);
        printf("  Z min,max,inc;    %+16.8f %+16.8f %+16.8f\n", 0.0, _dz * (_nz - 1), _dz);
    }

    /**
     * Notes
     * - User must have called setupDtOmegaInvQ_3D to initialize the array _dtOmegaInvQ
     * - wavefield arrays are switched in this call
     *     pCur -> pOld
     *     pOld -> pCur
     *     mCur -> mOld
     *     mOld -> mCur
     */
#if defined(__FUNCTION_CLONES__)
__attribute__((target_clones("avx","avx2","avx512f","default")))
#endif
    inline void timeStep() {

        applyFirstDerivatives3D_PlusHalf_Sandwich_Isotropic(
            _freeSurface, _nx, _ny, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDy, _invDz,
            _pCur, _pCur, _pCur, _b, _tmpPx1, _tmpPy1, _tmpPz1, _nbx, _nby, _nbz);

        applyFirstDerivatives3D_MinusHalf_TimeUpdate_Nonlinear_Isotropic(
            _freeSurface, _nx, _ny, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDy, _invDz, _dt,
            _tmpPx1, _tmpPy1, _tmpPz1, _v, _b, _dtOmegaInvQ, _pCur, _pSpace, _pOld, _nbx, _nby, _nbz);

        // swap pointers
        float *pswap = _pOld;
        _pOld = _pCur;
        _pCur = pswap;
    }

    /**
     * Scale spatial derivatives by v^2/b to make them temporal derivs
     */
#if defined(__FUNCTION_CLONES__)
__attribute__((target_clones("avx","avx2","avx512f","default")))
#endif
    inline void scaleSpatialDerivatives() {
#pragma omp parallel for collapse(3) num_threads(_nthread) schedule(static)
        for (long bx = 0; bx < _nx; bx += _nbx) {
            for (long by = 0; by < _ny; by += _nby) {
                for (long bz = 0; bz < _nz; bz += _nbz) {
                    const long kxmax = MIN(bx + _nbx, _nx);
                    const long kymax = MIN(by + _nby, _ny);
                    const long kzmax = MIN(bz + _nbz, _nz);

                    for (long kx = bx; kx < kxmax; kx++) {
                        for (long ky = by; ky < kymax; ky++) {
#pragma omp simd
                            for (long kz = bz; kz < kzmax; kz++) {
                                const long k = kx * _ny * _nz + ky * _nz + kz;
                                const float v2OverB = _v[k] * _v[k] / _b[k];
                                _pSpace[k] *= v2OverB;
                            }
                        }
                    }
                }
            }
        }
    }

    /**
     * Add the Born source for velocity only model-space at the current time
     *
     * User must have:
     *   - called the nonlinear forward
     *   - saved 2nd time derivative of pressure at corresponding time index in array dp2
     *   - Born source term will be injected into the _pCur array
     */
template<class Type>
#if defined(__FUNCTION_CLONES__)
__attribute__((target_clones("avx","avx2","avx512f","default")))
#endif
    inline void forwardBornInjection_V(Type *dVel, Type *wavefieldDP) {
#pragma omp parallel for collapse(3) num_threads(_nthread) schedule(static)
        for (long bx = 0; bx < _nx; bx += _nbx) {
            for (long by = 0; by < _ny; by += _nby) {
                for (long bz = 0; bz < _nz; bz += _nbz) {
                    const long kxmax = MIN(bx + _nbx, _nx);
                    const long kymax = MIN(by + _nby, _ny);
                    const long kzmax = MIN(bz + _nbz, _nz);

                    for (long kx = bx; kx < kxmax; kx++) {
                        for (long ky = by; ky < kymax; ky++) {
#pragma omp simd
                            for (long kz = bz; kz < kzmax; kz++) {
                                const long k = kx * _ny * _nz + ky * _nz + kz;

                                const Type V  = _v[k];
                                const Type B  = _b[k];
                                const Type dV = dVel[k];

                                const Type dt2v2OverB = _dt * _dt * V * V / B;
                                const Type factorV = 2 * B * dV / (V * V * V);

                                _pCur[k] += dt2v2OverB * wavefieldDP[k] * factorV;
                            }
                        }
                    }
                }
            }
        }
    }

    /**
     * Add the Born source for buoyancy and velocity model-space at the current time
     *
     * User must have:
     *   - called the nonlinear forward
     *   - saved 2nd time derivative of pressure at corresponding time index in array dp2
     *   - Born source term will be injected into the _pCur array
     *
     * TODO: if these second derivatice call and following loop is expensive, 
     *       could consider fusing the two derivative loops with the final loop
     */
template<class Type>
#if defined(__FUNCTION_CLONES__)
__attribute__((target_clones("avx","avx2","avx512f","default")))
#endif
    inline void forwardBornInjection_VB(Type *dVel, Type *dBuoy, Type *wavefieldP, Type *wavefieldDP) {

        applyFirstDerivatives3D_PlusHalf(
            _freeSurface, _nx, _ny, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDy, _invDz, 
            wavefieldP, wavefieldP, wavefieldP, _tmpPx1, _tmpPy1, _tmpPz1, _nbx, _nby, _nbz);

#pragma omp parallel for collapse(3) num_threads(_nthread) schedule(static)
        for (long bx = 0; bx < _nx; bx += _nbx) {
            for (long by = 0; by < _ny; by += _nby) {
                for (long bz = 0; bz < _nz; bz += _nbz) {
                    const long kxmax = MIN(bx + _nbx, _nx);
                    const long kymax = MIN(by + _nby, _ny);
                    const long kzmax = MIN(bz + _nbz, _nz);

                    for (long kx = bx; kx < kxmax; kx++) {
                        for (long ky = by; ky < kymax; ky++) {
#pragma omp simd
                            for (long kz = bz; kz < kzmax; kz++) {
                                const long k = kx * _ny * _nz + ky * _nz + kz;

                                const Type dB = dBuoy[k];

                                _tmpPx2[k] = dB * _tmpPx1[k];
                                _tmpPy2[k] = dB * _tmpPy1[k];
                                _tmpPz2[k] = dB * _tmpPz1[k];
                            }
                        }
                    }
                }
            }
        }

        applyFirstDerivatives3D_MinusHalf(
            _freeSurface, _nx, _ny, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDy, _invDz, 
            _tmpPx2, _tmpPy2, _tmpPz2, _tmpPx1, _tmpPy1, _tmpPz1, _nbx, _nby, _nbz);

#pragma omp parallel for collapse(3) num_threads(_nthread) schedule(static)
        for (long bx = 0; bx < _nx; bx += _nbx) {
            for (long by = 0; by < _ny; by += _nby) {
                for (long bz = 0; bz < _nz; bz += _nbz) {
                    const long kxmax = MIN(bx + _nbx, _nx);
                    const long kymax = MIN(by + _nby, _ny);
                    const long kzmax = MIN(bz + _nbz, _nz);

                    for (long kx = bx; kx < kxmax; kx++) {
                        for (long ky = by; ky < kymax; ky++) {
#pragma omp simd
                            for (long kz = bz; kz < kzmax; kz++) {
                                const long k = kx * _ny * _nz + ky * _nz + kz;

                                const Type V  = _v[k];
                                const Type B  = _b[k];
                                const Type dV = dVel[k];
                                const Type dB = dBuoy[k];

                                const Type dt2v2OverB = _dt * _dt * V * V / B;
                                const Type factorV = 2 * B * dV / (V * V * V);
                                const Type factorB = - dB / (V * V);

                                _pCur[k] += dt2v2OverB * (wavefieldDP[k] * (factorV + factorB) + _tmpPx1[k] + _tmpPy1[k] + _tmpPz1[k]);
                            }
                        }
                    }
                }
            }
        }
    }

    /**
     * Add the Born source for buoyancy only model-space at the current time
     *
     * User must have:
     *   - called the nonlinear forward
     *   - saved 2nd time derivative of pressure at corresponding time index in array dp2
     *   - Born source term will be injected into the _pCur array
     *
     * TODO: if these second derivatice call and following loop is expensive, 
     *       could consider fusing the two derivative loops with the final loop
     */
template<class Type>
#if defined(__FUNCTION_CLONES__)
__attribute__((target_clones("avx","avx2","avx512f","default")))
#endif
    inline void forwardBornInjection_B(Type *dBuoy, Type *wavefieldP, Type *wavefieldDP) {

        applyFirstDerivatives3D_PlusHalf(
            _freeSurface, _nx, _ny, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDy, _invDz, 
            wavefieldP, wavefieldP, wavefieldP, _tmpPx1, _tmpPy1, _tmpPz1, _nbx, _nby, _nbz);

#pragma omp parallel for collapse(3) num_threads(_nthread) schedule(static)
        for (long bx = 0; bx < _nx; bx += _nbx) {
            for (long by = 0; by < _ny; by += _nby) {
                for (long bz = 0; bz < _nz; bz += _nbz) {
                    const long kxmax = MIN(bx + _nbx, _nx);
                    const long kymax = MIN(by + _nby, _ny);
                    const long kzmax = MIN(bz + _nbz, _nz);

                    for (long kx = bx; kx < kxmax; kx++) {
                        for (long ky = by; ky < kymax; ky++) {
#pragma omp simd
                            for (long kz = bz; kz < kzmax; kz++) {
                                const long k = kx * _ny * _nz + ky * _nz + kz;

                                const Type dB = dBuoy[k];

                                _tmpPx2[k] = dB * _tmpPx1[k];
                                _tmpPy2[k] = dB * _tmpPy1[k];
                                _tmpPz2[k] = dB * _tmpPz1[k];
                            }
                        }
                    }
                }
            }
        }

        applyFirstDerivatives3D_MinusHalf(
            _freeSurface, _nx, _ny, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDy, _invDz, 
            _tmpPx2, _tmpPy2, _tmpPz2, _tmpPx1, _tmpPy1, _tmpPz1, _nbx, _nby, _nbz);

#pragma omp parallel for collapse(3) num_threads(_nthread) schedule(static)
        for (long bx = 0; bx < _nx; bx += _nbx) {
            for (long by = 0; by < _ny; by += _nby) {
                for (long bz = 0; bz < _nz; bz += _nbz) {
                    const long kxmax = MIN(bx + _nbx, _nx);
                    const long kymax = MIN(by + _nby, _ny);
                    const long kzmax = MIN(bz + _nbz, _nz);

                    for (long kx = bx; kx < kxmax; kx++) {
                        for (long ky = by; ky < kymax; ky++) {
#pragma omp simd
                            for (long kz = bz; kz < kzmax; kz++) {
                                const long k = kx * _ny * _nz + ky * _nz + kz;

                                const Type V  = _v[k];
                                const Type B  = _b[k];
                                const Type dB = dBuoy[k];

                                const Type dt2v2OverB = _dt * _dt * V * V / B;
                                const Type factorB = - dB / (V * V);

                                _pCur[k] += dt2v2OverB * (wavefieldDP[k] * factorB + _tmpPx1[k] + _tmpPy1[k] + _tmpPz1[k]);
                            }
                        }
                    }
                }
            }
        }
    }

    /**
     * Accumulate the Born image term at the current time for velocity only model-space
     *
     * User must have:
     *   - called the nonlinear forward
     *   - saved 2nd time derivative of pressure at corresponding time index in array dp2
     *   - Born image term will be accumulated iu the _dm array
     *
     *   - velocity term: [+ 2B/V^3 LtP r                                 ]
     *   - buoyancy  term: [- 1/V^2  LtP r - dx' P dx - dy' P dy - dz' P dz]
     *
     * TODO: if these adjoint accumulations are expensive, could consider fusing the 
     *       two derivative loops with the final loop
     */
template<class Type>
#if defined(__FUNCTION_CLONES__)
__attribute__((target_clones("avx","avx2","avx512f","default")))
#endif
    inline void adjointBornAccumulation_V(Type *dVel, Type *wavefieldDP) {

#pragma omp parallel for collapse(3) num_threads(_nthread) schedule(static)
        for (long bx = 0; bx < _nx; bx += _nbx) {
            for (long by = 0; by < _ny; by += _nby) {
                for (long bz = 0; bz < _nz; bz += _nbz) {
                    const long kxmax = MIN(bx + _nbx, _nx);
                    const long kymax = MIN(by + _nby, _ny);
                    const long kzmax = MIN(bz + _nbz, _nz);

                    for (long kx = bx; kx < kxmax; kx++) {
                        for (long ky = by; ky < kymax; ky++) {
#pragma omp simd
                            for (long kz = bz; kz < kzmax; kz++) {
                                const long k = kx * _ny * _nz + ky * _nz + kz;

                                const Type V = _v[k];
                                const Type B = _b[k];

                                const Type factorV = + 2 * B / (V * V * V);

                                dVel[k]  += factorV * wavefieldDP[k] * _pOld[k];
                            }
                        }
                    }
                }
            }
        }
    }

    /**
     * Accumulate the Born image term at the current time for velocity and buoyancy model-space
     *
     * User must have:
     *   - called the nonlinear forward
     *   - saved 2nd time derivative of pressure at corresponding time index in array dp2
     *   - Born image term will be accumulated iu the _dm array
     *
     *   - velocity term: [+ 2B/V^3 LtP r                                 ]
     *   - buoyancy  term: [- 1/V^2  LtP r - dx' P dx - dy' P dy - dz' P dz]
     */
template<class Type>
#if defined(__FUNCTION_CLONES__)
__attribute__((target_clones("avx","avx2","avx512f","default")))
#endif
    inline void adjointBornAccumulation_VB(Type *dVel, Type *dBuoy, Type *wavefieldP, Type *wavefieldDP) {
        
        applyFirstDerivatives3D_PlusHalf(
            _freeSurface, _nx, _ny, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDy, _invDz, 
            wavefieldP, wavefieldP, wavefieldP, _tmpPx1, _tmpPy1, _tmpPz1, _nbx, _nby, _nbz);

        applyFirstDerivatives3D_PlusHalf(
            _freeSurface, _nx, _ny, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDy, _invDz, 
            _pOld, _pOld, _pOld, _tmpPx2, _tmpPy2, _tmpPz2, _nbx, _nby, _nbz);

#pragma omp parallel for collapse(3) num_threads(_nthread) schedule(static)
        for (long bx = 0; bx < _nx; bx += _nbx) {
            for (long by = 0; by < _ny; by += _nby) {
                for (long bz = 0; bz < _nz; bz += _nbz) {
                    const long kxmax = MIN(bx + _nbx, _nx);
                    const long kymax = MIN(by + _nby, _ny);
                    const long kzmax = MIN(bz + _nbz, _nz);

                    for (long kx = bx; kx < kxmax; kx++) {
                        for (long ky = by; ky < kymax; ky++) {
#pragma omp simd
                            for (long kz = bz; kz < kzmax; kz++) {
                                const long k = kx * _ny * _nz + ky * _nz + kz;

                                const Type V = _v[k];
                                const Type B = _b[k];

                                const Type factorV = + 2 * B / (V * V * V);
                                const Type factorB = - 1 / (V * V);

                                dVel[k]  += factorV * wavefieldDP[k] * _pOld[k];
                                dBuoy[k] += factorB * wavefieldDP[k] * _pOld[k] 
                                    - _tmpPx1[k] * _tmpPx2[k] - _tmpPy1[k] * _tmpPy2[k] - _tmpPz1[k] * _tmpPz2[k];
                            }
                        }
                    }
                }
            }
        }
    }

    /**
     * Accumulate the Born image term at the current time for buoyancy only model-space
     *
     * User must have:
     *   - called the nonlinear forward
     *   - saved 2nd time derivative of pressure at corresponding time index in array dp2
     *   - Born image term will be accumulated iu the _dm array
     *
     *   - velocity term: [+ 2B/V^3 LtP r                                 ]
     *   - buoyancy  term: [- 1/V^2  LtP r - dx' P dx - dy' P dy - dz' P dz]
     */
template<class Type>
#if defined(__FUNCTION_CLONES__)
__attribute__((target_clones("avx","avx2","avx512f","default")))
#endif
    inline void adjointBornAccumulation_B(Type *dBuoy, Type *wavefieldP, Type *wavefieldDP) {

        applyFirstDerivatives3D_PlusHalf(
            _freeSurface, _nx, _ny, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDy, _invDz, 
            wavefieldP, wavefieldP, wavefieldP, _tmpPx1, _tmpPy1, _tmpPz1, _nbx, _nby, _nbz);

        applyFirstDerivatives3D_PlusHalf(
            _freeSurface, _nx, _ny, _nz, _nthread, _c8_1, _c8_2, _c8_3, _c8_4, _invDx, _invDy, _invDz, 
            _pOld, _pOld, _pOld, _tmpPx2, _tmpPy2, _tmpPz2, _nbx, _nby, _nbz);

#pragma omp parallel for collapse(3) num_threads(_nthread) schedule(static)
        for (long bx = 0; bx < _nx; bx += _nbx) {
            for (long by = 0; by < _ny; by += _nby) {
                for (long bz = 0; bz < _nz; bz += _nbz) {
                    const long kxmax = MIN(bx + _nbx, _nx);
                    const long kymax = MIN(by + _nby, _ny);
                    const long kzmax = MIN(bz + _nbz, _nz);

                    for (long kx = bx; kx < kxmax; kx++) {
                        for (long ky = by; ky < kymax; ky++) {
#pragma omp simd
                            for (long kz = bz; kz < kzmax; kz++) {
                                const long k = kx * _ny * _nz + ky * _nz + kz;

                                const Type V = _v[k];
                                const Type B = _b[k];

                                const Type factorB = - 1 / (V * V);

                                dBuoy[k] += factorB * wavefieldDP[k] * _pOld[k] 
                                    - _tmpPx1[k] * _tmpPx2[k] - _tmpPy1[k] * _tmpPy2[k] - _tmpPz1[k] * _tmpPz2[k];
                            }
                        }
                    }
                }
            }
        }
    }

    /**
     * Apply Kz wavenumber filter for up/down wavefield seperation
     * Faqi, 2011, Geophysics https://library.seg.org/doi/full/10.1190/1.3533914
     * 
     * We handle the FWI and RTM imaging conditions with a condition inside the OMP loop
     * 
     * Example Kz filtering with 8 samples 
     * frequency | +0 | +1 | +2 | +3 |  N | -3 | -2 | -1 |
     * original  |  0 |  1 |  2 |  3 |  4 |  5 |  6 |  7 |
     * upgoing   |  0 |  X |  X |  X |  4 |  5 |  6 |  7 |
     * dngoing   |  0 |  1 |  2 |  3 |  4 |  X |  X |  X |
     */
#if defined(__FUNCTION_CLONES__)
__attribute__((target_clones("avx","avx2","avx512f","default")))
#endif
    inline void adjointBornAccumulation_wavefieldsep(float *dVel, float *wavefieldDP, const long isFWI) {
        const long nfft = 2 * _nz;
        const float scale = 1.0f / (float)(nfft);

        // FWI: adj wavefield is dngoing
        // RTM: adj wavefield is upgoing
        const long kfft_adj = (isFWI) ? 0 : nfft / 2;

        std::complex<float> * __restrict__ tmp = new std::complex<float>[nfft];

        fftwf_plan planForward = fftwf_plan_dft_1d(nfft,
		    reinterpret_cast<fftwf_complex*>(tmp),
		    reinterpret_cast<fftwf_complex*>(tmp), +1, FFTW_ESTIMATE);

		fftwf_plan planInverse = fftwf_plan_dft_1d(nfft,
		    reinterpret_cast<fftwf_complex*>(tmp),
		    reinterpret_cast<fftwf_complex*>(tmp), -1, FFTW_ESTIMATE);

        delete [] tmp;

#pragma omp parallel num_threads(_nthread)
        {
            std::complex<float> * __restrict__ tmp_nlf = new std::complex<float>[nfft];
            std::complex<float> * __restrict__ tmp_adj = new std::complex<float>[nfft];

#pragma omp for  collapse(2) schedule(static)
            for (long bx = 0; bx < _nx; bx += _nbx) {
                for (long by = 0; by < _ny; by += _nby) {
                    const long kxmax = MIN(bx + _nbx, _nx);
                    const long kymax = MIN(by + _nby, _ny);

                    for (long kx = bx; kx < kxmax; kx++) {
                        for (long ky = by; ky < kymax; ky++) {

#pragma omp simd
                            for (long kfft = 0; kfft < nfft; kfft++) {
                                tmp_nlf[kfft] = 0;
                                tmp_adj[kfft] = 0;
                            }  

#pragma omp simd
                            for (long kz = 0; kz < _nz; kz++) {
                                const long k = kx * _ny * _nz + ky * _nz + kz;
                                tmp_nlf[kz] = scale * wavefieldDP[k];
                                tmp_adj[kz] = scale * _pOld[k];
                            }  

                            fftwf_execute_dft(planForward,
                                reinterpret_cast<fftwf_complex*>(tmp_nlf),
                                reinterpret_cast<fftwf_complex*>(tmp_nlf));

                            fftwf_execute_dft(planForward,
                                reinterpret_cast<fftwf_complex*>(tmp_adj),
                                reinterpret_cast<fftwf_complex*>(tmp_adj));

                            // upgoing: zero the positive frequencies, excluding Nyquist
                            // dngoing: zero the negative frequencies, excluding Nyquist
#pragma omp simd
                            for (long k = 1; k < nfft / 2; k++) {
                                tmp_nlf[nfft / 2 + k] = 0;
                                tmp_adj[kfft_adj + k] = 0;
                            }

                            fftwf_execute_dft(planInverse,
                                reinterpret_cast<fftwf_complex*>(tmp_nlf),
                                reinterpret_cast<fftwf_complex*>(tmp_nlf));

                            fftwf_execute_dft(planInverse,
                                reinterpret_cast<fftwf_complex*>(tmp_adj),
                                reinterpret_cast<fftwf_complex*>(tmp_adj));

                            // Faqi eq 10
                            // Applied to FWI: [Sup * Rdn]
                            // Applied to RTM: [Sup * Rup]
                            for (long kz = 0; kz < _nz; kz++) {
                                const long k = kx * _ny * _nz + ky * _nz + kz;
                                const float V = _v[k];
                                const float B = _b[k];
                                const float factor = 2 * B / (V * V * V);
                                dVel[k] += factor * real(tmp_nlf[kz] * tmp_adj[kz]);
                            }
                        } // end loop over ky
                    } // end loop over kx
                } // end loop over by
            } // end loop over bx
            
            delete [] tmp_nlf;
            delete [] tmp_adj;
        } // end parallel region

        fftwf_destroy_plan(planForward);
        fftwf_destroy_plan(planInverse);
    }

    template<class Type>
#if defined(__FUNCTION_CLONES__)
__attribute__((target_clones("avx","avx2","avx512f","default")))
#endif
    static inline void applyFirstDerivatives3D_PlusHalf_Sandwich_Isotropic(
            const long freeSurface,
            const long nx,
            const long ny,
            const long nz,
            const long nthread,
            const Type c8_1,
            const Type c8_2,
            const Type c8_3,
            const Type c8_4,
            const Type invDx,
            const Type invDy,
            const Type invDz,
            const Type * __restrict__ const inPX,
            const Type * __restrict__ const inPY,
            const Type * __restrict__ const inPZ,
            const Type * __restrict__ const fieldBuoy,
            Type * __restrict__ tmpPX,
            Type * __restrict__ tmpPY,
            Type * __restrict__ tmpPZ,
            const long BX_3D,
            const long BY_3D,
            const long BZ_3D) {

        const long nx4 = nx - 4;
        const long ny4 = ny - 4;
        const long nz4 = nz - 4;
        const long nynz = ny * nz;

        // zero output array: note only the annulus that is in the absorbing boundary needs to be zeroed
        for (long k = 0; k < 4; k++) {

#pragma omp parallel for num_threads(nthread) schedule(static)
            for (long kx = 0; kx < nx; kx++) {
#pragma omp simd
                for (long ky = 0; ky < ny; ky++) {
                    const long kindex1 = kx * ny * nz + ky * nz + k;
                    const long kindex2 = kx * ny * nz + ky * nz + (nz - 1 - k);
                    tmpPX[kindex1] = tmpPX[kindex2] = 0;
                    tmpPY[kindex1] = tmpPY[kindex2] = 0;
                    tmpPZ[kindex1] = tmpPZ[kindex2] = 0;
                }
            }

#pragma omp parallel for num_threads(nthread) schedule(static)
            for (long kx = 0; kx < nx; kx++) {
#pragma omp simd
                for (long kz = 0; kz < nz; kz++) {
                    const long kindex1 = kx * ny * nz + k * nz + kz;
                    const long kindex2 = kx * ny * nz + (ny - 1 - k) * nz + kz;
                    tmpPX[kindex1] = tmpPX[kindex2] = 0;
                    tmpPY[kindex1] = tmpPY[kindex2] = 0;
                    tmpPZ[kindex1] = tmpPZ[kindex2] = 0;
                }
            }

#pragma omp parallel for num_threads(nthread) schedule(static)
            for (long ky = 0; ky < ny; ky++) {
#pragma omp simd
                for (long kz = 0; kz < nz; kz++) {
                    const long kindex1 = k * ny * nz + ky * nz + kz;
                    const long kindex2 = (nx - 1 - k) * ny * nz + ky * nz + kz;
                    tmpPX[kindex1] = tmpPX[kindex2] = 0;
                    tmpPY[kindex1] = tmpPY[kindex2] = 0;
                    tmpPZ[kindex1] = tmpPZ[kindex2] = 0;
                }
            }
        }

        // interior
#pragma omp parallel for collapse(3) num_threads(nthread) schedule(static)
        for (long bx = 4; bx < nx4; bx += BX_3D) {
            for (long by = 4; by < ny4; by += BY_3D) {
                for (long bz = 4; bz < nz4; bz += BZ_3D) {
                    const long kxmax = MIN(bx + BX_3D, nx4);
                    const long kymax = MIN(by + BY_3D, ny4);
                    const long kzmax = MIN(bz + BZ_3D, nz4);

                    for (long kx = bx; kx < kxmax; kx++) {
                        const long kxnynz = kx * nynz;

                        for (long ky = by; ky < kymax; ky++) {
                            const long kynz = ky * nz;
                            const long kxnynz_kynz = kxnynz + kynz;

#pragma omp simd
                            for (long kz = bz; kz < kzmax; kz++) {
                                const long k = kxnynz_kynz + kz;
                                const long kynz_kz = + kynz + kz;

                                const Type stencilDPx =
                                    c8_1 * (- inPX[(kx+0) * nynz + kynz_kz] + inPX[(kx+1) * nynz + kynz_kz]) +
                                    c8_2 * (- inPX[(kx-1) * nynz + kynz_kz] + inPX[(kx+2) * nynz + kynz_kz]) +
                                    c8_3 * (- inPX[(kx-2) * nynz + kynz_kz] + inPX[(kx+3) * nynz + kynz_kz]) +
                                    c8_4 * (- inPX[(kx-3) * nynz + kynz_kz] + inPX[(kx+4) * nynz + kynz_kz]);

                                    const Type stencilDPy =
                                        c8_1 * (- inPY[kxnynz + (ky+0) * nz + kz] + inPY[kxnynz + (ky+1) * nz + kz]) +
                                        c8_2 * (- inPY[kxnynz + (ky-1) * nz + kz] + inPY[kxnynz + (ky+2) * nz + kz]) +
                                        c8_3 * (- inPY[kxnynz + (ky-2) * nz + kz] + inPY[kxnynz + (ky+3) * nz + kz]) +
                                        c8_4 * (- inPY[kxnynz + (ky-3) * nz + kz] + inPY[kxnynz + (ky+4) * nz + kz]);

                                    const Type stencilDPz =
                                        c8_1 * (- inPZ[kxnynz_kynz + (kz+0)] + inPZ[kxnynz_kynz + (kz+1)]) +
                                        c8_2 * (- inPZ[kxnynz_kynz + (kz-1)] + inPZ[kxnynz_kynz + (kz+2)]) +
                                        c8_3 * (- inPZ[kxnynz_kynz + (kz-2)] + inPZ[kxnynz_kynz + (kz+3)]) +
                                        c8_4 * (- inPZ[kxnynz_kynz + (kz-3)] + inPZ[kxnynz_kynz + (kz+4)]);

                                    const Type dPx = invDx * stencilDPx;
                                    const Type dPy = invDy * stencilDPy;
                                    const Type dPz = invDz * stencilDPz;

                                    const Type B = fieldBuoy[k];

                                    tmpPX[k] = B * dPx;
                                    tmpPY[k] = B * dPy;
                                    tmpPZ[k] = B * dPz;
                                }
                            }
                        }
                    }
                }
            }

            // roll on free surface
            if (freeSurface) {
#pragma omp parallel for num_threads(nthread) schedule(static)
                for (long kx = 4; kx < nx4; kx++) {
                    const long kxnynz = kx * nynz;

#pragma omp simd
                for (long ky = 4; ky < ny4; ky++) {
                    const long kynz = ky * nz;
                    const long kxnynz_kynz = kxnynz + kynz;

                    // kz = 0 -- 1/2 cells below free surface for Z derivative, at free surface for X/Y derivative
                    // X and Y derivatives are identically zero
                    // [kxnynz_kynz + 0]
                    {
                        const Type stencilDPz0 =
                            c8_1 * (- inPZ[kxnynz_kynz + 0] + inPZ[kxnynz_kynz + 1]) +
                            c8_2 * (+ inPZ[kxnynz_kynz + 1] + inPZ[kxnynz_kynz + 2]) +
                            c8_3 * (+ inPZ[kxnynz_kynz + 2] + inPZ[kxnynz_kynz + 3]) +
                            c8_4 * (+ inPZ[kxnynz_kynz + 3] + inPZ[kxnynz_kynz + 4]);

                        const Type dPx = 0;
                        const Type dPy = 0;
                        const Type dPz = invDz * stencilDPz0;

                        const long k = kxnynz_kynz + 0;

                        const Type B = fieldBuoy[k];

                        tmpPX[k] = B * dPx;
                        tmpPY[k] = B * dPy;
                        tmpPZ[k] = B * dPz;
                    }

                    // kz = 1 -- 1 1/2 cells below free surface for Z derivative, 1 cells below for X/Y derivative
                    // [kxnynz_kynz + 1]
                    {
                        const Type stencilDPx1 =
                            c8_1 * (- inPX[(kx+0) * nynz + kynz + 1] + inPX[(kx+1) * nynz + kynz + 1]) +
                            c8_2 * (- inPX[(kx-1) * nynz + kynz + 1] + inPX[(kx+2) * nynz + kynz + 1]) +
                            c8_3 * (- inPX[(kx-2) * nynz + kynz + 1] + inPX[(kx+3) * nynz + kynz + 1]) +
                            c8_4 * (- inPX[(kx-3) * nynz + kynz + 1] + inPX[(kx+4) * nynz + kynz + 1]);

                        const Type stencilDPy1 =
                            c8_1 * (- inPY[kxnynz + (ky+0) * nz + 1] + inPY[kxnynz + (ky+1) * nz + 1]) +
                            c8_2 * (- inPY[kxnynz + (ky-1) * nz + 1] + inPY[kxnynz + (ky+2) * nz + 1]) +
                            c8_3 * (- inPY[kxnynz + (ky-2) * nz + 1] + inPY[kxnynz + (ky+3) * nz + 1]) +
                            c8_4 * (- inPY[kxnynz + (ky-3) * nz + 1] + inPY[kxnynz + (ky+4) * nz + 1]);

                        const Type stencilDPz1 =
                            c8_1 * (- inPZ[kxnynz_kynz + 1] + inPZ[kxnynz_kynz + 2]) +
                            c8_2 * (- inPZ[kxnynz_kynz + 0] + inPZ[kxnynz_kynz + 3]) +
                            c8_3 * (+ inPZ[kxnynz_kynz + 1] + inPZ[kxnynz_kynz + 4]) +
                            c8_4 * (+ inPZ[kxnynz_kynz + 2] + inPZ[kxnynz_kynz + 5]);

                        const Type dPx = invDx * stencilDPx1;
                        const Type dPy = invDy * stencilDPy1;
                        const Type dPz = invDz * stencilDPz1;

                        const long k = kxnynz_kynz + 1;

                        const Type B = fieldBuoy[k];

                        tmpPX[k] = B * dPx;
                        tmpPY[k] = B * dPy;
                        tmpPZ[k] = B * dPz;
                    }

                    // kz = 2 -- 2 1/2 cells below free surface for Z derivative, 2 cells below for X/Y derivative
                    // [kxnynz_kynz + 2]
                    {
                        const Type stencilDPx2 =
                            c8_1 * (- inPX[(kx+0) * nynz + kynz + 2] + inPX[(kx+1) * nynz + kynz + 2]) +
                            c8_2 * (- inPX[(kx-1) * nynz + kynz + 2] + inPX[(kx+2) * nynz + kynz + 2]) +
                            c8_3 * (- inPX[(kx-2) * nynz + kynz + 2] + inPX[(kx+3) * nynz + kynz + 2]) +
                            c8_4 * (- inPX[(kx-3) * nynz + kynz + 2] + inPX[(kx+4) * nynz + kynz + 2]);

                        const Type stencilDPy2 =
                            c8_1 * (- inPY[kxnynz + (ky+0) * nz + 2] + inPY[kxnynz + (ky+1) * nz + 2]) +
                            c8_2 * (- inPY[kxnynz + (ky-1) * nz + 2] + inPY[kxnynz + (ky+2) * nz + 2]) +
                            c8_3 * (- inPY[kxnynz + (ky-2) * nz + 2] + inPY[kxnynz + (ky+3) * nz + 2]) +
                            c8_4 * (- inPY[kxnynz + (ky-3) * nz + 2] + inPY[kxnynz + (ky+4) * nz + 2]);

                        const Type stencilDPz2 =
                            c8_1 * (- inPZ[kxnynz_kynz + 2] + inPZ[kxnynz_kynz + 3]) +
                            c8_2 * (- inPZ[kxnynz_kynz + 1] + inPZ[kxnynz_kynz + 4]) +
                            c8_3 * (- inPZ[kxnynz_kynz + 0] + inPZ[kxnynz_kynz + 5]) +
                            c8_4 * (+ inPZ[kxnynz_kynz + 1] + inPZ[kxnynz_kynz + 6]);

                        const Type dPx = invDx * stencilDPx2;
                        const Type dPy = invDy * stencilDPy2;
                        const Type dPz = invDz * stencilDPz2;

                        const long k = kxnynz_kynz + 2;

                        const Type B = fieldBuoy[k];

                        tmpPX[k] = B * dPx;
                        tmpPY[k] = B * dPy;
                        tmpPZ[k] = B * dPz;
                    }

                // kz = 3 -- 3 1/2 cells below free surface for Z derivative, 3 cells below for X/Y derivative
                // [kxnynz_kynz + 3]
                    {
                        const Type stencilDPx3 =
                            c8_1 * (- inPX[(kx+0) * nynz + kynz + 3] + inPX[(kx+1) * nynz + kynz + 3]) +
                            c8_2 * (- inPX[(kx-1) * nynz + kynz + 3] + inPX[(kx+2) * nynz + kynz + 3]) +
                            c8_3 * (- inPX[(kx-2) * nynz + kynz + 3] + inPX[(kx+3) * nynz + kynz + 3]) +
                            c8_4 * (- inPX[(kx-3) * nynz + kynz + 3] + inPX[(kx+4) * nynz + kynz + 3]);

                        const Type stencilDPy3 =
                            c8_1 * (- inPY[kxnynz + (ky+0) * nz + 3] + inPY[kxnynz + (ky+1) * nz + 3]) +
                            c8_2 * (- inPY[kxnynz + (ky-1) * nz + 3] + inPY[kxnynz + (ky+2) * nz + 3]) +
                            c8_3 * (- inPY[kxnynz + (ky-2) * nz + 3] + inPY[kxnynz + (ky+3) * nz + 3]) +
                            c8_4 * (- inPY[kxnynz + (ky-3) * nz + 3] + inPY[kxnynz + (ky+4) * nz + 3]);

                        const Type stencilDPz3 =
                            c8_1 * (- inPZ[kxnynz_kynz + 3] + inPZ[kxnynz_kynz + 4]) +
                            c8_2 * (- inPZ[kxnynz_kynz + 2] + inPZ[kxnynz_kynz + 5]) +
                            c8_3 * (- inPZ[kxnynz_kynz + 1] + inPZ[kxnynz_kynz + 6]) +
                            c8_4 * (- inPZ[kxnynz_kynz + 0] + inPZ[kxnynz_kynz + 7]);

                        const Type dPx = invDx * stencilDPx3;
                        const Type dPy = invDy * stencilDPy3;
                        const Type dPz = invDz * stencilDPz3;

                        const long k = kxnynz_kynz + 3;

                        const Type B = fieldBuoy[k];

                        tmpPX[k] = B * dPx;
                        tmpPY[k] = B * dPy;
                        tmpPZ[k] = B * dPz;
                    }
                }
            }
        }
    }

    template<class Type>
#if defined(__FUNCTION_CLONES__)
__attribute__((target_clones("avx","avx2","avx512f","default")))
#endif
    static inline void applyFirstDerivatives3D_MinusHalf_TimeUpdate_Nonlinear_Isotropic(
            const long freeSurface,
            const long nx,
            const long ny,
            const long nz,
            const long nthread,
            const Type c8_1,
            const Type c8_2,
            const Type c8_3,
            const Type c8_4,
            const Type invDx,
            const Type invDy,
            const Type invDz,
            const Type dtMod,
            const Type * __restrict__ const tmpPX,
            const Type * __restrict__ const tmpPY,
            const Type * __restrict__ const tmpPZ,
            const Type * __restrict__ const fieldVel,
            const Type * __restrict__ const fieldBuoy,
            const Type * __restrict__ const dtOmegaInvQ,
            const Type * __restrict__ const pCur,
            Type * __restrict__ tmpPout,
            Type * __restrict__ pOld,
            const long BX_3D,
            const long BY_3D,
            const long BZ_3D) {

        const long nx4 = nx - 4;
        const long ny4 = ny - 4;
        const long nz4 = nz - 4;
        const long nynz = ny * nz;
        const Type dt2 = dtMod * dtMod;

        // zero output array: note only the annulus that is in the absorbing boundary needs to be zeroed
        for (long k = 0; k < 4; k++) {

#pragma omp parallel for num_threads(nthread) schedule(static)
            for (long kx = 0; kx < nx; kx++) {
#pragma omp simd
                for (long ky = 0; ky < ny; ky++) {
                    const long kindex1 = kx * ny * nz + ky * nz + k;
                    const long kindex2 = kx * ny * nz + ky * nz + (nz - 1 - k);
                    tmpPout[kindex1] = tmpPout[kindex2] = 0;
                }
            }

#pragma omp parallel for num_threads(nthread) schedule(static)
            for (long kx = 0; kx < nx; kx++) {
#pragma omp simd
                for (long kz = 0; kz < nz; kz++) {
                    const long kindex1 = kx * ny * nz + k * nz + kz;
                    const long kindex2 = kx * ny * nz + (ny - 1 - k) * nz + kz;
                    tmpPout[kindex1] = tmpPout[kindex2] = 0;
                }
            }

#pragma omp parallel for num_threads(nthread) schedule(static)
            for (long ky = 0; ky < ny; ky++) {
#pragma omp simd
                for (long kz = 0; kz < nz; kz++) {
                    const long kindex1 = k * ny * nz + ky * nz + kz;
                    const long kindex2 = (nx - 1 - k) * ny * nz + ky * nz + kz;
                    tmpPout[kindex1] = tmpPout[kindex2] = 0;
                }
            }

        }

        // interior
#pragma omp parallel for collapse(3) num_threads(nthread) schedule(static)
        for (long bx = 4; bx < nx4; bx += BX_3D) {
            for (long by = 4; by < ny4; by += BY_3D) {
                for (long bz = 4; bz < nz4; bz += BZ_3D) {
                    const long kxmax = MIN(bx + BX_3D, nx4);
                    const long kymax = MIN(by + BY_3D, ny4);
                    const long kzmax = MIN(bz + BZ_3D, nz4);

                    for (long kx = bx; kx < kxmax; kx++) {
                        const long kxnynz = kx * nynz;

                        for (long ky = by; ky < kymax; ky++) {
                            const long kynz = ky * nz;
                            const long kxnynz_kynz = kxnynz + kynz;

#pragma omp simd
                            for (long kz = bz; kz < kzmax; kz++) {
                                const long k = kxnynz_kynz + kz;
                                const long kynz_kz = + kynz + kz;

                                const Type stencilDPx =
                                    c8_1 * (- tmpPX[(kx-1) * nynz + kynz_kz] + tmpPX[(kx+0) * nynz + kynz_kz]) +
                                    c8_2 * (- tmpPX[(kx-2) * nynz + kynz_kz] + tmpPX[(kx+1) * nynz + kynz_kz]) +
                                    c8_3 * (- tmpPX[(kx-3) * nynz + kynz_kz] + tmpPX[(kx+2) * nynz + kynz_kz]) +
                                    c8_4 * (- tmpPX[(kx-4) * nynz + kynz_kz] + tmpPX[(kx+3) * nynz + kynz_kz]);

                                const Type stencilDPy =
                                    c8_1 * (- tmpPY[kxnynz + (ky-1) * nz + kz] + tmpPY[kxnynz + (ky+0) * nz + kz]) +
                                    c8_2 * (- tmpPY[kxnynz + (ky-2) * nz + kz] + tmpPY[kxnynz + (ky+1) * nz + kz]) +
                                    c8_3 * (- tmpPY[kxnynz + (ky-3) * nz + kz] + tmpPY[kxnynz + (ky+2) * nz + kz]) +
                                    c8_4 * (- tmpPY[kxnynz + (ky-4) * nz + kz] + tmpPY[kxnynz + (ky+3) * nz + kz]);

                                const Type stencilDPz =
                                    c8_1 * (- tmpPZ[kxnynz_kynz + (kz-1)] + tmpPZ[kxnynz_kynz + (kz+0)]) +
                                    c8_2 * (- tmpPZ[kxnynz_kynz + (kz-2)] + tmpPZ[kxnynz_kynz + (kz+1)]) +
                                    c8_3 * (- tmpPZ[kxnynz_kynz + (kz-3)] + tmpPZ[kxnynz_kynz + (kz+2)]) +
                                    c8_4 * (- tmpPZ[kxnynz_kynz + (kz-4)] + tmpPZ[kxnynz_kynz + (kz+3)]);

                                const Type dPx = invDx * stencilDPx;
                                const Type dPy = invDy * stencilDPy;
                                const Type dPz = invDz * stencilDPz;

                                const Type dt2V2_B = dt2 * fieldVel[k] * fieldVel[k] / fieldBuoy[k];

                                pOld[k] = dt2V2_B * (dPx + dPy + dPz) - dtOmegaInvQ[k] * (pCur[k] - pOld[k]) - pOld[k] + 2 * pCur[k];
                                tmpPout[k] = dPx + dPy + dPz;
                            }
                        }
                    }
                }
            }
        }

        // roll on free surface
        if (freeSurface) {
#pragma omp parallel for num_threads(nthread) schedule(static)
            for (long kx = 4; kx < nx4; kx++) {
                const long kxnynz = kx * nynz;

#pragma omp simd
                for (long ky = 4; ky < ny4; ky++) {
                    const long kynz = ky * nz;
                    const long kxnynz_kynz = kxnynz + kynz;

                    // kz = 0 -- at the free surface -- p = 0
                    // [kxnynz_kynz + 0]
                    {
                        const Type dPx = 0;
                        const Type dPy = 0;
                        const Type dPz = 0;

                        const long k = kxnynz_kynz + 0;
                        const Type dt2V2_B = dt2 * fieldVel[k] * fieldVel[k] / fieldBuoy[k];

                        pOld[k] = dt2V2_B * (dPx + dPy + dPz) - dtOmegaInvQ[k] * (pCur[k] - pOld[k]) - pOld[k] + 2 * pCur[k];
                        tmpPout[k] = dPx + dPy + dPz;
                    }

                    // kz = 1 -- one cell below the free surface
                    // [kxnynz_kynz + 1]
                    {
                        const Type stencilDPx1 =
                            c8_1 * (- tmpPX[(kx-1) * nynz + kynz + 1] + tmpPX[(kx+0) * nynz + kynz + 1]) +
                            c8_2 * (- tmpPX[(kx-2) * nynz + kynz + 1] + tmpPX[(kx+1) * nynz + kynz + 1]) +
                            c8_3 * (- tmpPX[(kx-3) * nynz + kynz + 1] + tmpPX[(kx+2) * nynz + kynz + 1]) +
                            c8_4 * (- tmpPX[(kx-4) * nynz + kynz + 1] + tmpPX[(kx+3) * nynz + kynz + 1]);

                        const Type stencilDPy1 =
                            c8_1 * (- tmpPY[kxnynz + (ky-1) * nz + 1] + tmpPY[kxnynz + (ky+0) * nz + 1]) +
                            c8_2 * (- tmpPY[kxnynz + (ky-2) * nz + 1] + tmpPY[kxnynz + (ky+1) * nz + 1]) +
                            c8_3 * (- tmpPY[kxnynz + (ky-3) * nz + 1] + tmpPY[kxnynz + (ky+2) * nz + 1]) +
                            c8_4 * (- tmpPY[kxnynz + (ky-4) * nz + 1] + tmpPY[kxnynz + (ky+3) * nz + 1]);

                        const Type stencilDPz1 =
                            c8_1 * (- tmpPZ[kxnynz_kynz + 0] + tmpPZ[kxnynz_kynz + 1]) +
                            c8_2 * (- tmpPZ[kxnynz_kynz + 0] + tmpPZ[kxnynz_kynz + 2]) +
                            c8_3 * (- tmpPZ[kxnynz_kynz + 1] + tmpPZ[kxnynz_kynz + 3]) +
                            c8_4 * (- tmpPZ[kxnynz_kynz + 2] + tmpPZ[kxnynz_kynz + 4]);

                        const Type dPx = invDx * stencilDPx1;
                        const Type dPy = invDy * stencilDPy1;
                        const Type dPz = invDz * stencilDPz1;

                        const long k = kxnynz_kynz + 1;
                        const Type dt2V2_B = dt2 * fieldVel[k] * fieldVel[k] / fieldBuoy[k];

                        pOld[k] = dt2V2_B * (dPx + dPy + dPz) - dtOmegaInvQ[k] * (pCur[k] - pOld[k]) - pOld[k] + 2 * pCur[k];
                        tmpPout[k] = dPx + dPy + dPz;
                    }

                    // kz = 2 -- two cells below the free surface
                    // [kxnynz_kynz + 2]
                    {
                        const Type stencilDPx2 =
                            c8_1 * (- tmpPX[(kx-1) * nynz + kynz + 2] + tmpPX[(kx+0) * nynz + kynz + 2]) +
                            c8_2 * (- tmpPX[(kx-2) * nynz + kynz + 2] + tmpPX[(kx+1) * nynz + kynz + 2]) +
                            c8_3 * (- tmpPX[(kx-3) * nynz + kynz + 2] + tmpPX[(kx+2) * nynz + kynz + 2]) +
                            c8_4 * (- tmpPX[(kx-4) * nynz + kynz + 2] + tmpPX[(kx+3) * nynz + kynz + 2]);

                        const Type stencilDPy2 =
                            c8_1 * (- tmpPY[kxnynz + (ky-1) * nz + 2] + tmpPY[kxnynz + (ky+0) * nz + 2]) +
                            c8_2 * (- tmpPY[kxnynz + (ky-2) * nz + 2] + tmpPY[kxnynz + (ky+1) * nz + 2]) +
                            c8_3 * (- tmpPY[kxnynz + (ky-3) * nz + 2] + tmpPY[kxnynz + (ky+2) * nz + 2]) +
                            c8_4 * (- tmpPY[kxnynz + (ky-4) * nz + 2] + tmpPY[kxnynz + (ky+3) * nz + 2]);

                        const Type stencilDPz2 =
                            c8_1 * (- tmpPZ[kxnynz_kynz + 1] + tmpPZ[kxnynz_kynz + 2]) +
                            c8_2 * (- tmpPZ[kxnynz_kynz + 0] + tmpPZ[kxnynz_kynz + 3]) +
                            c8_3 * (- tmpPZ[kxnynz_kynz + 0] + tmpPZ[kxnynz_kynz + 4]) +
                            c8_4 * (- tmpPZ[kxnynz_kynz + 1] + tmpPZ[kxnynz_kynz + 5]);

                        const Type dPx = invDx * stencilDPx2;
                        const Type dPy = invDy * stencilDPy2;
                        const Type dPz = invDz * stencilDPz2;

                        const long k = kxnynz_kynz + 2;
                        const Type dt2V2_B = dt2 * fieldVel[k] * fieldVel[k] / fieldBuoy[k];

                        pOld[k] = dt2V2_B * (dPx + dPy + dPz) - dtOmegaInvQ[k] * (pCur[k] - pOld[k]) - pOld[k] + 2 * pCur[k];
                        tmpPout[k] = dPx + dPy + dPz;
                    }

                    // kz = 3 -- three cells below the free surface
                    // [kxnynz_kynz + 3]
                    {
                        const Type stencilDPx3 =
                            c8_1 * (- tmpPX[(kx-1) * nynz + kynz + 3] + tmpPX[(kx+0) * nynz + kynz + 3]) +
                            c8_2 * (- tmpPX[(kx-2) * nynz + kynz + 3] + tmpPX[(kx+1) * nynz + kynz + 3]) +
                            c8_3 * (- tmpPX[(kx-3) * nynz + kynz + 3] + tmpPX[(kx+2) * nynz + kynz + 3]) +
                            c8_4 * (- tmpPX[(kx-4) * nynz + kynz + 3] + tmpPX[(kx+3) * nynz + kynz + 3]);

                        const Type stencilDPy3 =
                            c8_1 * (- tmpPY[kxnynz + (ky-1) * nz + 3] + tmpPY[kxnynz + (ky+0) * nz + 3]) +
                            c8_2 * (- tmpPY[kxnynz + (ky-2) * nz + 3] + tmpPY[kxnynz + (ky+1) * nz + 3]) +
                            c8_3 * (- tmpPY[kxnynz + (ky-3) * nz + 3] + tmpPY[kxnynz + (ky+2) * nz + 3]) +
                            c8_4 * (- tmpPY[kxnynz + (ky-4) * nz + 3] + tmpPY[kxnynz + (ky+3) * nz + 3]);

                        const Type stencilDPz3 =
                            c8_1 * (- tmpPZ[kxnynz_kynz + 2] + tmpPZ[kxnynz_kynz + 3]) +
                            c8_2 * (- tmpPZ[kxnynz_kynz + 1] + tmpPZ[kxnynz_kynz + 4]) +
                            c8_3 * (- tmpPZ[kxnynz_kynz + 0] + tmpPZ[kxnynz_kynz + 5]) +
                            c8_4 * (- tmpPZ[kxnynz_kynz + 0] + tmpPZ[kxnynz_kynz + 6]);

                        const Type dPx = invDx * stencilDPx3;
                        const Type dPy = invDy * stencilDPy3;
                        const Type dPz = invDz * stencilDPz3;

                        const long k = kxnynz_kynz + 3;
                        const Type dt2V2_B = dt2 * fieldVel[k] * fieldVel[k] / fieldBuoy[k];

                        tmpPout[k] = dPx + dPy + dPz;
                        pOld[k] = dt2V2_B * (dPx + dPy + dPz) - dtOmegaInvQ[k] * (pCur[k] - pOld[k]) - pOld[k] + 2 * pCur[k];
                    }
                }
            }
        }
    }

};

#endif
