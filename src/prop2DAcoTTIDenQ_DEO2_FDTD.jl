struct Prop2DAcoTTIDenQ_DEO2_FDTD
    p::Ptr{Cvoid}
end

function Prop2DAcoTTIDenQ_DEO2_FDTD(;
        nz=0,
        nx=0,
        nsponge=60,
        nbz=512,
        nbx=8,
        dz=5.0,
        dx=5.0,
        dt=0.001,
        nthreads=Sys.CPU_THREADS,
        freesurface=true,
        freqQ=5.0,
        qMin=0.1,
        qInterior=100.0)
    nz,nx,nsponge,nbz,nbx,nthreads = map(x->round(Int,x), (nz,nx,nsponge,nbz,nbx,nthreads))
    dz,dx,dt = map(x->Float32(x), (dz,dx,dt))

    @assert nx > 0
    @assert nz > 0
    @assert nsponge > 0
    @assert nthreads > 0
    @assert nbx > 0
    @assert nbz > 0

    fs = freesurface ? 1 : 0

    p = ccall((:Prop2DAcoTTIDenQ_DEO2_FDTD_alloc, libprop2DAcoTTIDenQ_DEO2_FDTD),
        Ptr{Cvoid},
        (Clong, Clong,    Clong, Clong, Clong,   Cfloat, Cfloat, Cfloat, Clong, Clong),
         fs,    nthreads, nx,    nz,    nsponge, dx,     dz,     dt,     nbx,   nbz)

    ccall((:Prop2DAcoTTIDenQ_DEO2_FDTD_SetupDtOmegaInvQ, libprop2DAcoTTIDenQ_DEO2_FDTD),
        Cvoid,
        (Ptr{Cvoid}, Cfloat, Cfloat, Cfloat),
         p,          freqQ,  qMin,   qInterior)

    Prop2DAcoTTIDenQ_DEO2_FDTD(p)
end

free(prop::Prop2DAcoTTIDenQ_DEO2_FDTD) = ccall((:Prop2DAcoTTIDenQ_DEO2_FDTD_free, libprop2DAcoTTIDenQ_DEO2_FDTD), Cvoid, (Ptr{Cvoid},), prop.p)

function size(prop::Prop2DAcoTTIDenQ_DEO2_FDTD)
    nz = ccall((:Prop2DAcoTTIDenQ_DEO2_FDTD_getNz, libprop2DAcoTTIDenQ_DEO2_FDTD), (Clong), (Ptr{Cvoid},), prop.p)
    nx = ccall((:Prop2DAcoTTIDenQ_DEO2_FDTD_getNx, libprop2DAcoTTIDenQ_DEO2_FDTD), (Clong), (Ptr{Cvoid},), prop.p)
    (nz,nx)
end

for _f in (:V, :Eps, :Eta, :B, :F, :PSpace, :MSpace, :PCur, :POld, :MCur, :MOld, :SinTheta, :CosTheta, :DtOmegaInvQ)
    symf = "Prop2DAcoTTIDenQ_DEO2_FDTD_get" * string(_f)
    @eval $(_f)(prop::Prop2DAcoTTIDenQ_DEO2_FDTD) = unsafe_wrap(Array, ccall(($symf, libprop2DAcoTTIDenQ_DEO2_FDTD), Ptr{Float32}, (Ptr{Cvoid},), prop.p), size(prop), own=false)
end

propagateforward!(prop::Prop2DAcoTTIDenQ_DEO2_FDTD) = ccall((:Prop2DAcoTTIDenQ_DEO2_FDTD_TimeStep, libprop2DAcoTTIDenQ_DEO2_FDTD), Cvoid, (Ptr{Cvoid},), prop.p)
propagateadjoint!(prop::Prop2DAcoTTIDenQ_DEO2_FDTD) = propagateforward!(prop) # self-adjoint

scale_spatial_derivatives!(prop::Prop2DAcoTTIDenQ_DEO2_FDTD) =
    ccall((:Prop2DAcoTTIDenQ_DEO2_FDTD_ScaleSpatialDerivatives, libprop2DAcoTTIDenQ_DEO2_FDTD), Cvoid, (Ptr{Cvoid},), prop.p)

abstract type Prop2DAcoTTIDenQ_DEO2_FDTD_Model end
struct Prop2DAcoTTIDenQ_DEO2_FDTD_Model_VEA <: Prop2DAcoTTIDenQ_DEO2_FDTD_Model end
struct Prop2DAcoTTIDenQ_DEO2_FDTD_Model_V <: Prop2DAcoTTIDenQ_DEO2_FDTD_Model end

# v,ϵ and η in model-space
function forwardBornInjection!(prop::Prop2DAcoTTIDenQ_DEO2_FDTD, modeltype::Prop2DAcoTTIDenQ_DEO2_FDTD_Model_VEA, dmodel, wavefields)
    ccall((:Prop2DAcoTTIDenQ_DEO2_FDTD_ForwardBornInjection_VEA, libprop2DAcoTTIDenQ_DEO2_FDTD), Cvoid,
        (Ptr{Cvoid}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat},        Ptr{Cfloat},        Ptr{Cfloat},          Ptr{Cfloat}),
         prop.p,     dmodel["v"], dmodel["ϵ"], dmodel["η"], wavefields["pold"], wavefields["mold"], wavefields["pspace"], wavefields["mspace"])
end

function adjointBornAccumulation!(prop::Prop2DAcoTTIDenQ_DEO2_FDTD, modeltype::Prop2DAcoTTIDenQ_DEO2_FDTD_Model_VEA,
        imagingcondition::ImagingConditionStandard, dmodel, wavefields)
    ccall((:Prop2DAcoTTIDenQ_DEO2_FDTD_AdjointBornAccumulation_VEA, libprop2DAcoTTIDenQ_DEO2_FDTD), Cvoid,
        (Ptr{Cvoid}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat},        Ptr{Cfloat},        Ptr{Cfloat},          Ptr{Cfloat}),
         prop.p,     dmodel["v"], dmodel["ϵ"], dmodel["η"], wavefields["pold"], wavefields["mold"], wavefields["pspace"], wavefields["mspace"])
end

# v in model-space
function forwardBornInjection!(prop::Prop2DAcoTTIDenQ_DEO2_FDTD, modeltype::Prop2DAcoTTIDenQ_DEO2_FDTD_Model_V, dmodel, wavefields)
    ccall((:Prop2DAcoTTIDenQ_DEO2_FDTD_ForwardBornInjection_V, libprop2DAcoTTIDenQ_DEO2_FDTD), Cvoid,
        (Ptr{Cvoid}, Ptr{Cfloat}, Ptr{Cfloat},          Ptr{Cfloat}),
         prop.p,     dmodel["v"], wavefields["pspace"], wavefields["mspace"])
end

function adjointBornAccumulation!(prop::Prop2DAcoTTIDenQ_DEO2_FDTD, modeltype::Prop2DAcoTTIDenQ_DEO2_FDTD_Model_V, 
        imagingcondition::ImagingConditionStandard, dmodel, wavefields)
    ccall((:Prop2DAcoTTIDenQ_DEO2_FDTD_AdjointBornAccumulation_V, libprop2DAcoTTIDenQ_DEO2_FDTD), Cvoid,
        (Ptr{Cvoid}, Ptr{Cfloat}, Ptr{Cfloat},          Ptr{Cfloat}),
         prop.p,     dmodel["v"], wavefields["pspace"], wavefields["mspace"])
end

function adjointBornAccumulation!(prop::Prop2DAcoTTIDenQ_DEO2_FDTD, modeltype::Prop2DAcoTTIDenQ_DEO2_FDTD_Model_V, 
        imagingcondition::ImagingConditionWaveFieldSeparationFWI, dmodel, wavefields)
    ccall((:Prop2DAcoTTIDenQ_DEO2_FDTD_AdjointBornAccumulation_wavefieldsep_V, libprop2DAcoTTIDenQ_DEO2_FDTD), Cvoid,
        (Ptr{Cvoid}, Ptr{Cfloat}, Ptr{Cfloat},          Ptr{Cfloat},          Clong),
         prop.p,     dmodel["v"], wavefields["pspace"], wavefields["mspace"], 1)
end

function adjointBornAccumulation!(prop::Prop2DAcoTTIDenQ_DEO2_FDTD, modeltype::Prop2DAcoTTIDenQ_DEO2_FDTD_Model_V, 
        imagingcondition::ImagingConditionWaveFieldSeparationRTM, dmodel, wavefields)
    ccall((:Prop2DAcoTTIDenQ_DEO2_FDTD_AdjointBornAccumulation_wavefieldsep_V, libprop2DAcoTTIDenQ_DEO2_FDTD), Cvoid,
        (Ptr{Cvoid}, Ptr{Cfloat}, Ptr{Cfloat},          Ptr{Cfloat},          Clong),
         prop.p,     dmodel["v"], wavefields["pspace"], wavefields["mspace"], 0)
end

function show(io::IO, prop::Prop2DAcoTTIDenQ_DEO2_FDTD)
    nx = ccall((:Prop2DAcoTTIDenQ_DEO2_FDTD_getNx, libprop2DAcoTTIDenQ_DEO2_FDTD), (Clong), (Ptr{Cvoid},), prop.p)
    nz = ccall((:Prop2DAcoTTIDenQ_DEO2_FDTD_getNz, libprop2DAcoTTIDenQ_DEO2_FDTD), (Clong), (Ptr{Cvoid},), prop.p)
    write(io, "Prop2DAcoTTIDenQ_DEO2_FDTD -- nx,nz; $nx,$nz")
end
