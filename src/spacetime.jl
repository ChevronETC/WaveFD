#
# time utility functions
#
function default_dtmod_helper(stenciltype::Symbol, Q::Int, D::Int, mindel::T, velmax::T, dtmod::T, dtrec::T, alpha::T, ndim::Int) where T<:Real
    a = stencilcoeffs(:Fornberg, Q, D, ndim, dtmod, velmax, mindel, T)
    c = courantnumber(a, ndim)
    default_dtmod_helper_helper(mindel, velmax, c, dtmod, dtrec, alpha)
end
function default_dtmod_helper_helper(mindel::T, velmax::T, c::T, dtmod::T, dtrec::T, alpha::T) where T<:Real
    dtmod_auto = alpha*c*mindel/velmax
    if dtmod < 0
        dtmod = dtmod_auto
    end
    if 0.99*dtmod > dtmod_auto
        throw(ArgumentError("dtmod=$(dtmod), dtmod_auto=$(dtmod_auto), mindel=$(mindel), velmax=$(velmax)"))
    end
    T(dtrec/ceil(dtrec/min(dtmod,dtrec) - .1)) # tricky
end

"""
# 2D
    dtmod = WaveFD.default_dtmod(stenciltype, Q, D, dz, dx, velmax, dtmod, dtrec, alpha)
    dtmod = WaveFD.default_dtmod(alpha, dz, dx, velmax, dtrec, dtmod) # assumes courant=1 .. should go away once we become less lazy about doing analysis
"""
default_dtmod(stenciltype::Symbol, Q::Int, D::Int, dz::T, dx::T, velmax::T, dtmod::T, dtrec::T, alpha::T) where {T<:AbstractFloat} = default_dtmod_helper(stenciltype, Q, D, min(dz,dx), velmax, dtmod, dtrec, alpha, 2)
default_dtmod(alpha::T, dz::T, dx::T, velmax::T, dtrec::T, dtmod::T) where {T<:AbstractFloat} = default_dtmod_helper_helper(min(dz,dx), velmax, one(T), dtmod, dtrec, alpha)

"""
# 3D
    dtmod = WaveFD.default_dtmod(stenciltype, Q, D, dz, dy, dx, velmax, dtmod, dtrec, alpha)
    dtmod = WaveFD.default_dtmod(alpha, dz, dy, dx, velmax, dtrec, dtmod) # assumes courant=1, lazy option, rather than doing proper analysis

where `dtmod::AbstractFloat` is a stable finite-difference time step determined by the C-F-L condition (https://en.wikipedia.org/wiki/Courant%E2%80%93Friedrichs%E2%80%93Lewy_condition).
A modified Courant number is used for the C-F-L condition, and is determined by the choice of finite-difference stencil and a scaling factor `alpha`.  Making `alpha` smaller should reduce
numerical dispersion, but at the expense of a smaller `dtmod`.  We have found that setting `alpha=0.25` produces results with small dispersion.  For simplicity, we compute `dtmod`
such that `dtrec` is a scalar multiple of `dtmod`.

## paramters
* `proptype` choose betweeen `:O28_C`,`:O28_Julia`,`:O24_C`, or `:O24_Julia`.  The Courant number depends on the finite difference stencil.  In this case, either 4th or 8th order
* `stenciltype` choose between `:Fornberg` and `:Nihei`.  `:Fornberg` are standard coefficients derived from Taylor. `:Nihei` is an attempt at producing optimal coefficients but only work for constant earth models.
* `Q` is the accuracy order of the stencil, choose between `4` and `8`
* `D` is the derivative order of the stencil, choose between `1` and `2`. Note that `1` assumes staggered, and `2` assumes centered.
* `dz`,`dy`,`dx` are the grid spacing
* `velmax` expected maximum velocity in the earth model
* `dtmod` desired modeling time interval.  If `dtmod<0`, then `dtmod` will be computed according to C-F-L and the recording dt.  Otherwise, we will attempt to use the user specified `dtmod` and throw
an error if it violates C-F-L condition.  However, we will, if needed, reduce `dtmod` so that `dtrec` is a scalar multiple of `dtmod`.
* `dtrec` recording time interval
* `alpha` Courant number multiplier

## Example
```julia
dtmod=-1.0
dz,dx,velmax,dtrec,alpha=0.01,0.01,5.0,0.004,0.25
Q,D=8,1
dtmod = WaveFD.default_dtmod(:Fornberg, Q, D, dz, dx, velmax, dtmod, dtrec, alpha)
```

## Notes
This method is not exported, because it is not the expectation that the user will need to call this method unless they want/need to by-pass the logic in the
Jot Wavefield propagation operators.
"""
default_dtmod(stenciltype::Symbol, Q::Int, D::Int, dz::T, dy::T, dx::T, velmax::T, dtmod::T, dtrec::T, alpha::T) where {T<:AbstractFloat} = default_dtmod_helper(stenciltype, Q, D, min(dz,dy,dx), velmax, dtmod, dtrec, alpha, 3)
default_dtmod(alpha::T, dz::T, dy::T, dx::T, velmax::T, dtrec::T, dtmod::T) where {T<:AbstractFloat} = default_dtmod_helper_helper(min(dz,dy,dx), velmax, one(T), dtmod, dtrec, alpha)

"""
    it0, ntmod = WaveFD.default_ntmod(dtrec,dtmod,t0,ntrec)

`it0` the time index corresponding to zero-time, and `ntmod` the total number of time points, including negative time.
`dtrec`,`dtmod` and `ntrec` are `Real`, while `t0` can either be `Real`, `Array{Array{Real}}` or `DArray{Array{Real}}`.  If `t0` is
of type `Array{Real}`, then `it0` is determined using `minimum(t0)`.
"""
function default_ntmod(dtrec::Real, dtmod::Real, t0::Real, ntrec::Int)
    ntmod = round(Int,dtrec/dtmod)*(ntrec - 1) + 1
    it0 = max(round(Int64, -t0/dtmod) + 1, 1)
    ntmod += it0-1
    it0, ntmod
end
default_ntmod(dtrec::Real, dtmod::Real, t0::Array, ntrec::Int) = default_ntmod(dtrec, dtmod, mapreduce(i->minimum(t0[i]), min, 1:length(t0)), ntrec)

function default_ntmod(dtrec::Real, dtmod::Real, t0::DArray, ntrec::Int)
    pids = procs(t0)
    npids = length(pids)
    r = Array{Future}(undef, npids)
    @sync for (i,pid) in enumerate(pids)
        r[i] = @spawnat pid begin
            t0_local = localpart(t0)
            t0min_local = minimum(t0_local[1])
            for j = 2:length(t0_local)
                t0min_local = minimum(t0_local[j]) < t0min_local ? minimum(t0_local[j]) : t0min_local
            end
            t0min_local
        end
    end
    t0min = fetch(r[1])
    for i = 2:npids
        t0test = fetch(r[i])
        t0min = t0test < t0min ? t0test : t0min
    end
    default_ntmod(dtrec, dtmod, t0min, ntrec)
end

"""
    ntmod = default_ntmod(dtrec,dtmod,ntrec)

`ntmod` is the total number of time points.  It is assuming that the first time index corresponds to zero-time.
"""
default_ntmod(dtrec::Real, dtmod::Real, ntrec::Int) = default_ntmod(dtrec, dtmod, 0.0, ntrec)[2]



#
# time shift codes
#
struct TimeShift{T<:AbstractFloat}
    h::Vector{T}
    nshift::Int
    shift_fraction::T
    bc::String
end

"""
    h = WaveFD.shiftfilter(shift; [length=12, α=1.0, bc="zero"])

Build a cosine tapered sinc filter for shifting an array by a decimal number of samples. The optional parameters are:

* `length::Int` length of the filter (better to be an even number)
* `α::Real` parameter to control the strength of the cosine taper. α=0 -> no taper
* `bc::String` how to deal with the edges. Currently the two options are "zero" and "nearest".

# Notes
* It is checked that `0 <= α <= 1`.
* The taper is centered to the middle of the filter and goes all the way to the edges.
* For α = 1, the value of the taper at the edges is ≈ 0.
* The shift is split into integer number of samples and a residual fractional sample.
"""
function shiftfilter(shift::Real; length::Int=12, α::T=1.0, bc::String="zero") where {T<:AbstractFloat}
    @assert 0 <= α <= 1
    @assert bc ∈ ("zero","nearest")
    nshift = floor(Int, shift)
    shift_fraction::T = (shift - nshift)

    hl = div(length,2) + length % 2
    j = [1:length;] .- (div(length,2) + 1 )
    a = sinc.(j .+ shift_fraction) .* ((1-α) .+ α/2*(1 .+ cos.(π*(j .+ shift_fraction)/hl)))
    a ./= sum(a)

    TimeShift(a, nshift, shift_fraction, bc)
end

"""
    WaveFD.shiftforward!(h, d, m, bc)

Shift `m::Array{T,N}` to `d::Array{T,N}` by a decimal number of samples. `d` and `m` have the same size.
A positive shift will delay the trace. `h` is built using `WaveFD.shiftfilter`. Both filter and amount 
of shifting must be shorter than the arrays, otherwise, an assertion will fail. 
For examples:

    WaveFD.shiftforward!(WaveFD.shiftfilter(-3.3), d, m)
    WaveFD.shiftforward!(WaveFD.shiftfilter(13.34, length=12, α=1.0, bc="zero"), d, m)

Note that `N=1`, `N=2` or `N=3` are supported. If `N=2` or `N=3`, then interpolation is done along the fast dimension.
"""
function shiftforward!(H::TimeShift{<:AbstractFloat}, d::StridedArray{T,1}, m::StridedArray{T,1}) where {T<:Real}
    @assert length(m) == length(d)

    if H.nshift == 0 && H.shift_fraction ≈ 0
        d .= m
        return nothing
    end

    n = length(m)
    nshift = H.nshift
    nfilter = length(H.h)
    n_filter_over_2, n_filter_over_2_remainder  = divrem(nfilter,2)
    n_half_filter = n_filter_over_2 + n_filter_over_2_remainder
    bc_multiplier = (H.bc=="nearest") ? 1 : 0

    n > n_half_filter || error("filter length must be shorter than twice the signal length")
    nshift < n || error("the amount of shifting must be shorter than the total length of the signal") 

    # create a padded array
    mpad = Vector{T}(undef, n + 2*n_half_filter)
    mpad[1+n_half_filter:n+n_half_filter] .= m
    mpad[1:n_half_filter] .= bc_multiplier * m[1]
    mpad[n+n_half_filter+1:end] .= bc_multiplier * m[n]

    # shift the padded array by a fraction of a sample
    fill!(d, 0.0)
    if H.shift_fraction ≈ 0
        d[:] = mpad[1+n_half_filter:n+n_half_filter]
    else
        for i in 1:n
            for j in 1:nfilter
                d[i] += mpad[i+n_half_filter + j - n_filter_over_2 - 1] * H.h[j] 
            end
        end
    end

    # shift by an integer number of samples then extrapolate first or last sample
    if nshift > 0
        d[1+nshift:n] = d[1:n-nshift]
        d[1:nshift] .= bc_multiplier * d[1]
    elseif nshift<0
        d[1:n+nshift] = d[1-nshift:n]
        d[n+nshift+1:n] .= bc_multiplier * d[n]
    end
    nothing
end

"""
    WaveFD.shiftadjoint!(h, m, d, bc)

Adjoint of `WaveFD.shiftforward`. Depending on the filter, it is "almost" like a shifting with negative shift used in forward mode (inverse shifting). 
"""
function shiftadjoint!(H::TimeShift{<:AbstractFloat}, m::StridedArray{T,1}, d::StridedArray{T,1}) where {T<:Real}
    @assert length(m) == length(d)

    if H.nshift == 0 && H.shift_fraction ≈ 0
        m .= d
        return nothing
    end

    n = length(m)
    nshift = H.nshift
    nfilter = length(H.h)
    n_filter_over_2, n_filter_over_2_remainder  = divrem(nfilter,2)
    n_half_filter = n_filter_over_2 + n_filter_over_2_remainder
    bc_multiplier = (H.bc=="nearest") ? 1 : 0

    n > n_half_filter || error("filter length must be shorter than twice the signal length")
    nshift < n || error("the amount of shifting must be shorter than the total length of the signal")

    # shift by integer number of samples and extrapolate
    fill!(m,0)
    if nshift > 0
        m[1:n-nshift] = d[1+nshift:n]
        m[n-nshift] += nshift*bc_multiplier * d[n]
    elseif nshift<0
        m[1-nshift:n] = d[1:n+nshift]
        m[1-nshift] -= nshift*bc_multiplier * d[1]
    else
        m .= d
    end

    # create a padded array
    mpad = Vector{T}(undef, n + 2*n_half_filter)
    
    # shift by a fraction of a sample
    fill!(mpad, 0.0)
    if H.shift_fraction ≈ 0
        mpad[1+n_half_filter:n+n_half_filter] .= m
    else
        for i in 1:n
            for j in 1:nfilter
                mpad[i+n_half_filter + j - n_filter_over_2 - 1] += m[i] * H.h[j]
            end
        end
    end

    # finalize
    m[1:n] = mpad[1+n_half_filter:n+n_half_filter]
    m[1] += bc_multiplier*n_half_filter * mpad[1+n_half_filter]
    m[n] += bc_multiplier*n_half_filter * mpad[n+n_half_filter]
    nothing
end

function shiftforward!(H::TimeShift{<:AbstractFloat}, d::StridedArray{T,2}, m::StridedArray{T,2}) where {T<:Real}
    Threads.@threads for i2 = 1:size(m,2)
        d_trace = @view d[:,i2]
        m_trace = @view m[:,i2]
        shiftforward!(H, d_trace, m_trace)
    end
    nothing
end

function shiftforward!(H::TimeShift{<:AbstractFloat}, d::StridedArray{T,3}, m::StridedArray{T,3}) where {T<:Real}
    Threads.@threads for i3 = 1:size(m,3)
        for i2 = 1:size(m,2)
            d_trace = @view d[:,i2,i3]
            m_trace = @view m[:,i2,i3]
            shiftforward!(H, d_trace, m_trace)
        end
    end
    nothing
end

function shiftadjoint!(H::TimeShift{<:AbstractFloat}, m::StridedArray{T,2}, d::StridedArray{T,2}) where {T<:Real}
    Threads.@threads for i2 = 1:size(m,2)
        d_trace = @view d[:,i2]
        m_trace = @view m[:,i2]
        shiftadjoint!(H, m_trace, d_trace)
    end
    nothing
end

function shiftadjoint!(H::TimeShift{<:AbstractFloat}, m::StridedArray{T,3}, d::StridedArray{T,3}) where {T<:Real}
    Threads.@threads for i3 = 1:size(m,3)
        for i2 = 1:size(m,2)
            d_trace = @view d[:,i2,i3]
            m_trace = @view m[:,i2,i3]
            shiftadjoint!(H, m_trace, d_trace)
        end
    end
    nothing
end



#
# time interpolation codes
#

mutable struct TimeInterp{T<:AbstractFloat,C<:Language}
    h::Array{Array{T,1},1}
    nthreads::Int64
end

"""
    h = WaveFD.interpfilters(dtmod, dtrec [, mode=0, impl=WaveFD.LangC, nthreads=Sys.CPU_THREADS])

Build an 8 point sinc filter, mapping between `dtmod` and `dtrec` sampling.  The optional parameters are:

* `mode::Int` determines if amplitude is preserved in the forward (mode=0) or adjoint operation (mode=1)
* `impl::WaveFD.Language` can be set to either `WaveFD.LangJulia()` or `WaveFD.LangC()` to determine which code-path to follow
* `nthreads`  If `impl=WaveFD.LangC()`, then OMP is used for threading the 2D and 3D arrays where time is assumed to be along the fast dimension

# Notes
It is assumed that `dtmod<dtrec`:

* forward operator (see `WaveFD.interpforward!`) - interpolates from `dtmod` to `dtrec`, preserves amplitude if `mode=0`.  This is the default behaviour.
* adjoint operator (see `WaveFD.interpadjoint!`) - interpolates from `dtrec` to `dtmod`, preserves amplitude if `mode=1`
"""
function interpfilters(dtmod::T, dtrec::T, mode::Int=0, impl::Language=LangC(), nthreads=Sys.CPU_THREADS) where T<:Real
    @assert dtmod <= dtrec
    n = round(Int, dtrec/dtmod)
    if abs(n*dtmod - dtrec)/dtrec > 1e-6
        throw(ArgumentError("abs(n*dtmod-dtrec)/dtrec > 1e-6, n=$(n), dtrec=$(dtrec), dtmod=$(dtmod), abs(n*dtmod-dtrec)/dtrec=$(abs(n*dtmod-dtrec)/n)"))
    end

    d = dtmod/dtrec*[0:n-1;]
    fmax = T(0.066 + 0.265*log(8.0))
    j = [1:8;]

    a = @. sinc((j-1)*fmax)
    A = [a[1] a[2] a[3] a[4] a[5] a[6] a[7] a[8] ;
         a[2] a[1] a[2] a[3] a[4] a[5] a[6] a[7] ;
         a[3] a[2] a[1] a[2] a[3] a[4] a[5] a[6] ;
         a[4] a[3] a[2] a[1] a[2] a[3] a[4] a[5] ;
         a[5] a[4] a[3] a[2] a[1] a[2] a[3] a[4] ;
         a[6] a[5] a[4] a[3] a[2] a[1] a[2] a[3] ;
         a[7] a[6] a[5] a[4] a[3] a[2] a[1] a[2] ;
         a[8] a[7] a[6] a[5] a[4] a[3] a[2] a[1]]

    h = Array{T}[]
    for i = 1:n
        c = @. sinc((4-j+d[i])*fmax)
        push!(h, A\c)
        if mode == 0
            h[i] ./= n # denom is to make sure that the amplitude in the forward is correct
        end
    end

    TimeInterp{T,typeof(impl)}(h,nthreads)
end

"""
    WaveFD.interpadjoint!(h, m, d)

Interpolate from `d::Array{T,N}` (coarse) to `m::Array{T,N}` (fine) using the sinc filter coefficients in `h::Array{Array{T,1},1}`.  `h` is built using
`WaveFD.interpfilters`.  For example:

    WaveFD.interpadjoint!(WaveFD.interpfilters(.001,.004), m, d)

Note that we support, `N=1`, `N=2` or `N=3`. If `N=2` or `N=3`, then interpolation is done along the fast dimension.  By default, `interpadjoint!` does not preserve
amplitude (see `WaveFD.interpfilters`).
"""
function interpadjoint!(H::TimeInterp{T,LangJulia}, m::StridedArray{T,1}, d::StridedArray{T,1}) where T
    if length(m) == length(d)
        m .= d
        return nothing
    end
    h = H.h
    nh = length(h)
    nd = length(d)
    nm = length(m)
    fill!(m,0.0)

    interpadjoint_helper!(m,d,h,nm,nh,1,4,8) # i=1
    interpadjoint_helper!(m,d,h,nm,nh,2,3,8) # i=2
    interpadjoint_helper!(m,d,h,nm,nh,3,2,8) # i=3
    for i = 4:nd-4
        interpadjoint_helper!(m,d,h,nm,nh,i,1,8)
    end
    interpadjoint_helper!(m,d,h,nm,nh,nd-3,1,7) # i=nd-3
    interpadjoint_helper!(m,d,h,nm,nh,nd-2,1,6) # i=nd-2
    interpadjoint_helper!(m,d,h,nm,nh,nd-1,1,5) # i=nd-1
    nothing
end

@inline function interpadjoint_helper!(m,d,h,nm,nh,i,lo,hi)
    j = (i-1)*nh
    im4 = i-4
    for k=1:nh
        jk = j+k
        jk > nm && continue
        for l = lo:hi
            m[jk] += h[k][l] * d[im4+l]
        end
    end
end

"""
    WaveFD.interpforward!(h, d, m)

Interpolate from `m::Array{T,N}` (fine) to `d::Array{T,N}` (coarse) using the sinc filter coefficients in `h::Array{Array{T,1},1}`.  `h` is built using
`WaveFD.interpfilters`.  For example:

    WaveFD.interpadjoint!(WaveFD.interpfilters(.001,.004), m, d)

Note that we support, `N=1`, `N=2` or `N=3`. If `N=2` or `N=3`, then interpolation is done along the fast dimension.  By default, `interpforward!` preserves
amplitude (see `WaveFD.interpfilters!`).
"""
function interpforward!(H::TimeInterp{T,LangJulia}, d::StridedArray{T,1}, m::StridedArray{T,1}) where T
    if length(d) == length(m)
        d .= m
        return nothing
    end
    h = H.h
    nh = length(h)
    nd = length(d)
    nm = length(m)
    fill!(d,0.0)
    interpforward_helper!(d,m,h,nh,1,4,8) # i=1
    interpforward_helper!(d,m,h,nh,2,3,8) # i=2
    interpforward_helper!(d,m,h,nh,3,2,8) # i=3
    for i = 4:nd-4
        interpforward_helper!(d,m,h,nh,i,1,8)
    end
    interpforward_helper!(d,m,h,nh,nd-3,1,7) # i=nd-3
    interpforward_helper!(d,m,h,nh,nd-2,1,6) # i=nd-2
    interpforward_helper!(d,m,h,nh,nd-1,1,5) # i=nd-1
    nothing
end

@inline function interpforward_helper!(d,m,h,nh,i,lo,hi)
    j = (i-1)*nh
    im4 = i-4
    for k = 1:nh
        jk = j + k
        jk > length(m) && continue
        for l = lo:hi
            d[im4+l] += h[k][l]*m[jk]
        end
    end
end

function interpforward!(H::TimeInterp{T,LangJulia}, d::StridedArray{T,2}, m::StridedArray{T,2}) where T
    for i2 = 1:size(m,2)
        d_trace = @view d[:,i2]
        m_trace = @view m[:,i2]
        interpforward!(H, d_trace, m_trace)
    end
    nothing
end

function interpforward!(H::TimeInterp{T,LangJulia}, d::StridedArray{T,3}, m::StridedArray{T,3}) where T
    for i3 = 1:size(m,3), i2 = 1:size(m,2)
        d_trace = @view d[:,i2,i3]
        m_trace = @view m[:,i2,i3]
        interpforward!(H, d_trace, m_trace)
    end
    nothing
end

function interpadjoint!(H::TimeInterp{T,LangJulia}, m::StridedArray{T,2}, d::StridedArray{T,2}) where T
    for i2 = 1:size(m,2)
        d_trace = @view d[:,i2]
        m_trace = @view m[:,i2]
        interpadjoint!(H, m_trace, d_trace)
    end
    nothing
end

function interpadjoint!(H::TimeInterp{T,LangJulia}, m::StridedArray{T,3}, d::StridedArray{T,3}) where T
    for i3 = 1:size(m,3), i2=1:size(m,2)
        d_trace = @view d[:,i2,i3]
        m_trace = @view m[:,i2,i3]
        interpadjoint!(H, m_trace, d_trace)
    end
    nothing
end

#
# fast-path (OpenMP) codes for time interpolation (should be able to delete these once Julia gets shared memory parallel support)
#
function interpadjoint!(H::TimeInterp{Float32,LangC}, m::StridedArray{Float32,1}, d::StridedArray{Float32,1})
    if length(m) == length(d)
        m .= d
        return nothing
    end
    ccall(
        (:interpadjoint_1d_float, libspacetime),
        Ptr{Cvoid},
        (Ptr{Ptr{Cfloat}}, Ptr{Cfloat}, Ptr{Cfloat}, Csize_t,     Csize_t,   Csize_t),
         H.h,              m,           d,           length(H.h), length(m), length(d)
    )
end

function interpadjoint!(H::TimeInterp{Float64,LangC}, m::StridedArray{Float64,1}, d::StridedArray{Float64,1})
    if length(m) == length(d)
        m .= d
        return nothing
    end
    ccall(
        (:interpadjoint_1d_double, libspacetime),
        Ptr{Cvoid},
        (Ptr{Ptr{Cdouble}}, Ptr{Cdouble}, Ptr{Cdouble}, Csize_t,     Csize_t,   Csize_t),
         H.h,               m,            d,            length(H.h), length(m), length(d)
    )
end

function interpadjoint!(H::TimeInterp{Float32,LangC}, m::StridedArray{Float32,2}, d::StridedArray{Float32,2})
    if length(m) == length(d)
        m .= d
        return nothing
    end
    ccall(
        (:interpadjoint_nd_float, libspacetime),
        Ptr{Cvoid},
        (Ptr{Ptr{Cfloat}}, Ptr{Cfloat}, Ptr{Cfloat}, Csize_t,     Csize_t,   Csize_t,   Csize_t,   Csize_t),
         H.h,              m,           d,           length(H.h), size(m,1), size(d,1), size(d,2), H.nthreads
    )
    nothing
end

function interpadjoint!(H::TimeInterp{Float64,LangC}, m::StridedArray{Float64,2}, d::StridedArray{Float64,2})
    if length(m) == length(d)
        m .= d
        return nothing
    end
    ccall(
        (:interpadjoint_nd_double, libspacetime),
        Ptr{Cvoid},
        (Ptr{Ptr{Cdouble}}, Ptr{Cdouble}, Ptr{Cdouble}, Csize_t,     Csize_t,   Csize_t,   Csize_t,   Csize_t),
         H.h,               m,            d,            length(H.h), size(m,1), size(d,1), size(d,2), H.nthreads
    )
    nothing
end

function interpadjoint!(H::TimeInterp{Float32,LangC}, m::StridedArray{Float32,3}, d::StridedArray{Float32,3})
    if length(m) == length(d)
        m .= d
        return nothing
    end
    ccall(
        (:interpadjoint_nd_float, libspacetime),
        Ptr{Cvoid},
        (Ptr{Ptr{Cfloat}}, Ptr{Cfloat}, Ptr{Cfloat}, Csize_t,     Csize_t,   Csize_t,   Csize_t,             Csize_t),
         H.h,              m,           d,           length(H.h), size(m,1), size(d,1), size(d,2)*size(d,3), H.nthreads
    )
    nothing
end

function interpadjoint!(H::TimeInterp{Float64,LangC}, m::StridedArray{Float64,3}, d::StridedArray{Float64,3})
    if length(m) == length(d)
        m .= d
        return nothing
    end
    ccall(
        (:interpadjoint_nd_double, libspacetime),
        Ptr{Cvoid},
        (Ptr{Ptr{Cdouble}}, Ptr{Cdouble}, Ptr{Cdouble}, Csize_t,     Csize_t,   Csize_t,   Csize_t,             Csize_t),
         H.h,               m,            d,            length(H.h), size(m,1), size(d,1), size(d,2)*size(d,3), H.nthreads
    )
    nothing
end

function interpforward!(H::TimeInterp{Float32,LangC}, d::StridedArray{Float32,1}, m::StridedArray{Float32,1})
    if length(d) == length(m)
        d .= m
        return nothing
    end
    ccall(
        (:interpforward_1d_float, libspacetime),
        Ptr{Cvoid},
        (Ptr{Ptr{Cfloat}}, Ptr{Cfloat}, Ptr{Cfloat}, Csize_t,     Csize_t,   Csize_t),
         H.h,              d,           m,           length(H.h), length(d), length(m)
    )
    nothing
end

function interpforward!(H::TimeInterp{Float64,LangC}, d::StridedArray{Float64,1}, m::StridedArray{Float64,1})
    if length(d) == length(m)
        d .= m
        return nothing
    end
    ccall(
        (:interpforward_1d_double, libspacetime),
        Ptr{Cvoid},
        (Ptr{Ptr{Cdouble}}, Ptr{Cdouble}, Ptr{Cdouble}, Csize_t,     Csize_t,   Csize_t),
         H.h,               d,            m,            length(H.h), length(d), length(m)
    )
    nothing
end

function interpforward!(H::TimeInterp{Float32,LangC}, d::StridedArray{Float32,2}, m::StridedArray{Float32,2})
    if length(d) == length(m)
        d .= m
        return nothing
    end
    ccall(
        (:interpforward_nd_float, libspacetime),
        Ptr{Cvoid},
        (Ptr{Ptr{Cfloat}}, Ptr{Cfloat}, Ptr{Cfloat}, Csize_t,      Csize_t,   Csize_t,   Csize_t,   Csize_t),
         H.h,              d,           m,           length(H.h),  size(d,1), size(m,1), size(m,2), H.nthreads
    )
    nothing
end

function interpforward!(H::TimeInterp{Float64,LangC}, d::StridedArray{Float64,2}, m::StridedArray{Float64,2})
    if length(d) == length(m)
        d .= m
        return nothing
    end
    ccall(
        (:interpforward_nd_double, libspacetime),
        Ptr{Cvoid},
        (Ptr{Ptr{Cdouble}}, Ptr{Cdouble}, Ptr{Cdouble}, Csize_t,     Csize_t,   Csize_t,   Csize_t,   Csize_t),
         H.h,               d,            m,            length(H.h), size(d,1), size(m,1), size(m,2), H.nthreads
    )
    nothing
end

function interpforward!(H::TimeInterp{Float32,LangC}, d::StridedArray{Float32,3}, m::StridedArray{Float32,3})
    if length(d) == length(m)
        d .= m
        return nothing
    end
    ccall(
        (:interpforward_nd_float, libspacetime),
        Ptr{Cvoid},
        (Ptr{Ptr{Cfloat}}, Ptr{Cfloat}, Ptr{Cfloat}, Csize_t,      Csize_t,   Csize_t,   Csize_t,             Csize_t),
         H.h,               d,           m,           length(H.h), size(d,1), size(m,1), size(m,2)*size(m,3), H.nthreads
    )
    nothing
end

function interpforward!(H::TimeInterp{Float64,LangC}, d::StridedArray{Float64,3}, m::StridedArray{Float64,3})
    if length(d) == length(m)
        d .= m
        return nothing
    end
    ccall(
        (:interpforward_nd_double, libspacetime),
        Ptr{Cvoid},
        (Ptr{Ptr{Cdouble}}, Ptr{Cdouble}, Ptr{Cdouble}, Csize_t,     Csize_t,   Csize_t,   Csize_t,             Csize_t),
         H.h,               d,            m,            length(H.h), size(d,1), size(m,1), size(m,2)*size(m,3), H.nthreads
    )
    nothing
end

#
# space interpolation codes
#
#
# Graham Hicks' Kaiser windowed sinc function method, Geophysics 67(1), P. 156-166
# paper suggests r=4, with b=6.31 for kmax=pi/2 and with b=4.14 for kmax=2pi/3
#
# For our problem we are using 5 grid points per wave-length --
# k_max = w_max / c_min = 2 pi fmax / c_min
# k_nyq = pi / del = pi / (c_min / (5fmax)) = 5 fmax * pi / c_min
#
# k_max   2 pi fmax / c_min   2   4        1   5
# ----- = ----------------- = - = --    <  - = --
# k_nyq   5 pi fmax / c_min   5   10       2   10
#
# Hicks suggests b=6.42 for kmax=1/2q
kaiser(b::T, r::Int64, x::T) where {T<:Union{Float32,Float64}} = abs(x) > r ? zero(T) : T(besseli(0,b*sqrt(1-(x/r)^2))/besseli(0,b))

function hicks!(d::AbstractMatrix, b::Real, r::Integer, alphaz::Real, alphax::Real, fz::Vector, fx::Vector)
    l = 2*r
    @inbounds for i = 1:l
        a = i - r
        z = a - alphaz
        x = a - alphax
        fz[i] = kaiser(b,r,z)*sinc(z)
        fx[i] = kaiser(b,r,x)*sinc(x)
    end
    @inbounds for j = 1:l, i = 1:l
        d[i,j] = fz[i]*fx[j]
    end
end

function hicks(b::Real, r::Integer, alphaz::Real, alphax::Real)
    l = 2*r
    d = Array{eltype(b)}(undef, l, l)
    fz = Array{eltype(b)}(undef, l)
    fx = Array{eltype(b)}(undef, l)
    hicks!(d, b, r, alphaz, alphax, fz, fx)
    d
end

function hicks!(d::AbstractArray{T,3}, b::Real, r::Integer, alphaz::Real, alphay::Real, alphax::Real, fz::Vector, fy::Vector, fx::Vector) where T<:Real
    l = 2*r
    @inbounds for i = 1:l
        a = i - r
        z = a - alphaz
        y = a - alphay
        x = a - alphax
        fz[i] = kaiser(b,r,z)*sinc(z)
        fy[i] = kaiser(b,r,y)*sinc(y)
        fx[i] = kaiser(b,r,x)*sinc(x)
    end
    @inbounds for k = 1:l, j = 1:l, i = 1:l
        d[i,j,k] = fz[i]*fy[j]*fx[k]
    end
end

function hicks(b::Real, r::Integer, alphaz::Real, alphay::Real, alphax::Real)
    l = 2*r
    d = Array{eltype(b)}(undef, l, l, l)
    fz = Array{eltype(b)}(undef, l)
    fy = Array{eltype(b)}(undef, l)
    fx = Array{eltype(b)}(undef, l)
    hicks!(d, b, r, alphaz, alphay, alphax, fz, fy, fx)
    d
end

function delete_out_of_bounds_points!(iz, ix, ir, c, nz, nx)
    out_of_bounds_indices = Int[]
    for i in eachindex(iz)
        if !( (1 <= iz[i] <= nz) && (1 <= ix[i] <= nx) )
            push!(out_of_bounds_indices, i)
        end
    end

    if length(out_of_bounds_indices) > 0
        @warn "deleting out-of-bounds indices, $(100*length(out_of_bounds_indices)/length(c))% of total."
        for i in (iz, ix, ir, c)
            deleteat!(i, out_of_bounds_indices)
        end
    end
end

function delete_out_of_bounds_points!(iz, iy, ix, ir, c, nz, ny, nx)
    out_of_bounds_indices = Int[]
    for i in eachindex(iz)
        if !( (1 <= iz[i] <= nz) && (1 <= iy[i] <= ny) && (1 <= ix[i] <= nx) )
            push!(out_of_bounds_indices, i)
        end
    end

    if length(out_of_bounds_indices) > 0
        @warn "deleting out-of bounds indices, $(100*length(out_of_bounds_indices)/length(c))% of total."
        for i in (iz, iy, ix, ir, c)
            deleteat!(i, out_of_bounds_indices)
        end
    end
end

function delete_zero_coefficients_points!(iz, ix, ir, c::Vector{T}) where {T}
    zero_coefficient_indices = Int[]
    for i in eachindex(iz)
        if abs(c[i]) < eps(T)
            push!(zero_coefficient_indices, i)
        end
    end

    if length(zero_coefficient_indices) > 0
        for i in (iz, ix, ir, c)
            deleteat!(i, zero_coefficient_indices)
        end
    end
end

function delete_zero_coefficients_points!(iz, iy, ix, ir, c::Vector{T}) where {T}
    zero_coefficient_indices = Int[]
    for i in eachindex(iz)
        if abs(c[i]) < eps(T)
            push!(zero_coefficient_indices, i)
        end
    end

    if length(zero_coefficient_indices) > 0
        for i in (iz, iy, ix, ir, c)
            deleteat!(i, zero_coefficient_indices)
        end
    end
end

abstract type SourcePoint end

struct SourcePoint32 <: SourcePoint
    iu::Clong # linear index into 3D or 2D array
    ir::Clong # reciever index
    c::Cfloat # injection coefficient
end

struct SourcePoint64 <: SourcePoint
    iu::Clong # linear index into 3D or 2D array
    ir::Clong # reciever index
    c::Cdouble # injection coefficient
end

SourcePoint(iu,ir,c::Float32) = SourcePoint32(iu,ir,c)
SourcePoint(iu,ir,c::Float64) = SourcePoint64(iu,ir,c)
SourcePointType(_::Type{Float32}) = SourcePoint32
SourcePointType(_::Type{Float64}) = SourcePoint64

# Base.isless(x::SourcePoint, y::SourcePoint) = x.iu < y.iu

function hickscoeffs(dz::T, dx::T, z0::Float64, x0::Float64, nz::Int64, nx::Int64, z::Array{Float64,1}, x::Array{Float64,1}, fs_index=1, zstagger=0.0, xstagger=0.0) where T
    hicks_b = T(6.42)
    hicks_r = 4
    hicks_l = 2*hicks_r
    hicks_fz = Array{T}(undef, hicks_l)
    hicks_fx = Array{T}(undef, hicks_l)
    iz = Vector{Int}(undef, length(z)*hicks_l*hicks_l)
    ix = Vector{Int}(undef, length(z)*hicks_l*hicks_l)
    ir = Vector{Int}(undef, length(z)*hicks_l*hicks_l)
    c = Vector{T}(undef, length(z)*hicks_l*hicks_l)
    L = LinearIndices((hicks_l,hicks_l,length(z)))
    for i = 1:length(z)
        iz_mid = floor(Int, (z[i] - z0)/dz) + 1
        ix_mid = floor(Int, (x[i] - x0)/dx) + 1
        alphaz = T((z[i] - (z0 + (iz_mid - 1 + zstagger)*dz))/dz)
        alphax = T((x[i] - (x0 + (ix_mid - 1 + xstagger)*dx))/dx)

        i1 = L[1,1,i]
        i2 = L[hicks_l,hicks_l,i]
        ci = reshape(view(c,i1:i2), hicks_l, hicks_l)
        hicks!(ci, hicks_b, hicks_r, alphaz, alphax, hicks_fz, hicks_fx)

        for _ix=1:hicks_l, _iz=1:hicks_l
            j = L[_iz,_ix,i]
            iz[j] = iz_mid - hicks_r + _iz
            ix[j] = ix_mid - hicks_r + _ix
            ir[j] = i
        end

        # reflect source terms above the free-surface
        # we assume that if there is no free-surface, then
        # the model would have being padded accordingly, making
        # the below if statement always evaluate to false
        j = L[1,1,i]
        if iz[j] == fs_index - 3
            ci[5,:] -= ci[3,:]
            ci[6,:] -= ci[2,:]
            ci[7,:] -= ci[1,:]
        end
        if iz[j] == fs_index - 2
            ci[4,:] -= ci[2,:]
            ci[5,:] -= ci[1,:]
        end
        if iz[j] == fs_index - 1
            ci[3,:] -= ci[1,:]
        end
    end

    delete_out_of_bounds_points!(iz, ix, ir, c, nz, nx)
    delete_zero_coefficients_points!(iz, ix, ir, c)

    M = LinearIndices((nz,nx))
    [SourcePoint(M[iz[i],ix[i]], ir[i], c[i]) for i in eachindex(iz)]
end

function hickscoeffs(dz::T, dy::T, dx::T, z0::Float64, y0::Float64, x0::Float64, nz::Int64, ny::Int64, nx::Int64, z::Array{Float64,1}, y::Array{Float64,1}, x::Array{Float64,1}, fs_index=1, zstagger=0.0, ystagger=0.0, xstagger=0.0) where T
    hicks_b = T(6.42)
    hicks_r = 4
    hicks_l = 2*hicks_r
    hicks_fz = Array{T}(undef, hicks_l)
    hicks_fy = Array{T}(undef, hicks_l)
    hicks_fx = Array{T}(undef, hicks_l)
    iz = Vector{Int}(undef, length(z)*hicks_l*hicks_l*hicks_l)
    iy = Vector{Int}(undef, length(z)*hicks_l*hicks_l*hicks_l)
    ix = Vector{Int}(undef, length(z)*hicks_l*hicks_l*hicks_l)
    ir = Vector{Int}(undef, length(z)*hicks_l*hicks_l*hicks_l)
    c = Vector{T}(undef, length(z)*hicks_l*hicks_l*hicks_l)
    L = LinearIndices((hicks_l,hicks_l,hicks_l,length(z)))
    for i = 1:length(z)
        iz_mid = floor(Int, (z[i] - z0)/dz) + 1
        iy_mid = floor(Int, (y[i] - y0)/dy) + 1
        ix_mid = floor(Int, (x[i] - x0)/dx) + 1
        alphaz = T((z[i] - (z0 + (iz_mid - 1 + zstagger)*dz))/dz)
        alphay = T((y[i] - (y0 + (iy_mid - 1 + ystagger)*dy))/dy)
        alphax = T((x[i] - (x0 + (ix_mid - 1 + xstagger)*dx))/dx)

        i1 = L[1,1,1,i]
        i2 = L[hicks_l,hicks_l,hicks_l,i]
        ci = reshape(view(c,i1:i2), hicks_l, hicks_l, hicks_l)
        hicks!(ci, hicks_b, hicks_r, alphaz, alphay, alphax, hicks_fz, hicks_fy, hicks_fx)

        for _ix=1:hicks_l, _iy=1:hicks_l, _iz=1:hicks_l
            j = L[_iz,_iy,_ix,i]
            iz[j] = iz_mid - hicks_r + _iz
            iy[j] = iy_mid - hicks_r + _iy
            ix[j] = ix_mid - hicks_r + _ix
            ir[j] = i
        end

        # reflect source terms above the free-surface
        # we assume that if there is no free-surface, then
        # the model would have being padded accordingly, making
        # the below if statement always evaluate to false
        j = L[1,1,1,i]
        if iz[j] == fs_index-3
            ci[5,:,:] -= ci[3,:,:]
            ci[6,:,:] -= ci[2,:,:]
            ci[7,:,:] -= ci[1,:,:]
        end
        if iz[j] == fs_index-2
            ci[4,:,:] -= ci[2,:,:]
            ci[5,:,:] -= ci[1,:,:]
        end
        if iz[j] == fs_index-1
            ci[3,:,:] -= ci[1,:,:]
        end
    end

    delete_out_of_bounds_points!(iz, iy, ix, ir, c, nz, ny, nx)
    delete_zero_coefficients_points!(iz, iy, ix, ir, c)

    M = LinearIndices((nz,ny,nx))
    [SourcePoint(M[iz[i],iy[i],ix[i]], ir[i], c[i]) for i in eachindex(iz)]
end

function linearcoeffs(dz::T, dx::T, z0::Float64, x0::Float64, nz::Int64, nx::Int64, z::Array{Float64,1}, x::Array{Float64,1}, fs_index=1) where T
    iz = Vector{Int}(undef, length(z)*4)
    ix = Vector{Int}(undef, length(z)*4)
    ir = Vector{Int}(undef, length(z)*4)
    c = Vector{T}(undef, length(z)*4)
    L = LinearIndices((2,2,length(z)))
    for i = 1:length(z)
        kx = floor(Int64, (x[i] - x0) / dx) + 1
        kz = floor(Int64, (z[i] - z0) / dz) + 1

        rx = (x[i] - (x0 + dx*(kx-1))) / dx
        rz = (z[i] - (z0 + dz*(kz-1))) / dz

        c[L[1,1,i]] = (1.0 - rx) * (1.0 - rz)
        c[L[2,1,i]] = (1.0 - rx) *        rz
        c[L[1,2,i]] =        rx  * (1.0 - rz)
        c[L[2,2,i]] =        rx  *        rz

        for _ix=1:2, _iz=1:2
            j = L[_iz,_ix,i]
            iz[j] = kz + _iz - 1
            ix[j] = kx + _ix - 1
            ir[j] = i
        end

        # free surface should always be zero
        if iz[L[1,1,i]] == fs_index
            for _ix = 1:2
                c[L[1,_ix,i]] = 0
            end
        end
    end

    delete_out_of_bounds_points!(iz, ix, ir, c, nz, nx)
    delete_zero_coefficients_points!(iz, ix, ir, c)

    M = LinearIndices((nz,nx))
    [SourcePoint(M[iz[i],ix[i]], ir[i], c[i]) for i in eachindex(iz)]
end

function linearcoeffs(dz::T, dy::T, dx::T, z0::Float64, y0::Float64, x0::Float64, nz::Int64, ny::Int64, nx::Int64, z::Array{Float64,1}, y::Array{Float64,1}, x::Array{Float64,1}, fs_index=1) where T
    iz = Vector{Int}(undef, length(z)*8)
    iy = Vector{Int}(undef, length(z)*8)
    ix = Vector{Int}(undef, length(z)*8)
    ir = Vector{Int}(undef, length(z)*8)
    c = Vector{T}(undef, length(z)*8)
    L = LinearIndices((2,2,2,length(z)))
    for i in eachindex(z)
        kx = floor(Int64, (x[i] - x0) / dx) + 1
        ky = floor(Int64, (y[i] - y0) / dy) + 1
        kz = floor(Int64, (z[i] - z0) / dz) + 1

        rx = (x[i] - (x0 + dx*(kx-1))) / dx
        ry = (y[i] - (y0 + dy*(ky-1))) / dy
        rz = (z[i] - (z0 + dz*(kz-1))) / dz

        c[L[1,1,1,i]] = (1.0 - rx) * (1.0 - ry) * (1.0 - rz)
        c[L[2,1,1,i]] = (1.0 - rx) * (1.0 - ry) *        rz
        c[L[1,2,1,i]] = (1.0 - rx) *        ry  * (1.0 - rz)
        c[L[2,2,1,i]] = (1.0 - rx) *        ry  *        rz
        c[L[1,1,2,i]] =        rx  * (1.0 - ry) * (1.0 - rz)
        c[L[2,1,2,i]] =        rx  * (1.0 - ry) *        rz
        c[L[1,2,2,i]] =        rx  *        ry  * (1.0 - rz)
        c[L[2,2,2,i]] =        rx  *        ry  *        rz

        for _ix=1:2, _iy=1:2, _iz=1:2
            j = L[_iz,_iy,_ix,i] 
            iz[j] = kz + _iz - 1
            iy[j] = ky + _iy - 1
            ix[j] = kx + _ix - 1
            ir[j] = i
        end

        # free surface should always be zero
        if iz[L[1,1,1,i]] == fs_index
            for _ix = 1:2, _iy = 1:2
                c[L[1,_iy,_ix,i]] = 0
            end
        end
    end

    delete_out_of_bounds_points!(iz, iy, ix, ir, c, nz, ny, nx)
    delete_zero_coefficients_points!(iz, iy, ix, ir, c)

    M = LinearIndices((nz,ny,nx))
    [SourcePoint(M[iz[i],iy[i],ix[i]], ir[i], c[i]) for i in eachindex(iz)]
end

"""
    sourceblocks = source_blocking(points, nthreads=Sys.CPU_THREADS)

Blocks points `points::Vector{SourcePoint}` into partitions such that no two partitions have the same
grid-point (iz,ix).

# Inputs
* points are the injection points. Each point has structure (iu,ir,c) where,
  * iu is the linear index in the finite difference grid
  * ir is the receiver/source index
  * c is the injection/extraction coefficient
* nthreads are the number of threads that will be used and helps inform the number of partitions.

# Output
The output is organized as:
```
[
    [ # block 1
        (i1,ir1,c1), # first point in block 1
        (i2,ir2,c2)  # second point in block 1
        ...
    ],
    [ # block 2
        (i3,ir3,c3), # first point in block 2
        (i4,ir4,c4), # second point in block 2
        ...
    ],
    ...
]
```
and where i1,i2,... are the linearized indices of the finite difference grid

see also WaveFD.injectdata!
"""
source_blocking(points::Vector{T}, nthreads=Sys.CPU_THREADS) where T <: SourcePoint = blocking(points, x->x.iu, nthreads)

"""
receiverblocks = receiver_blocking(points, nthreads=Sys.CPU_THREADS)

Blocks points `points::Vector{SourcePoint}` into partitions such that no two partitions have the same
receiver index.

# Inputs
* points are the extraction points. Each point has structure (iu,ir,c) where,
  * iu is the linear index in the finite difference grid
  * ir is the receiver/source index
  * c is the injection/extraction coefficient
* nthreads are the number of threads that will be used and helps inform the number of partitions.

# Output
The output is organized as:
```
[
    [ # block 1
        (i1,ir1,c1), # first point in block 1
        (i2,ir2,c2)  # second point in block 1
        ...
    ],
    [ # block 2
        (i3,ir3,c3), # first point in block 2
        (i4,ir4,c4), # second point in block 2
        ...
    ],
    ...
]
```
and where i1,i2,... are the linearized indices of the finite difference grid

see also WaveFD.extractdata!
"""
receiver_blocking(points::Vector{T}, nthreads=Sys.CPU_THREADS) where T <:SourcePoint = blocking(points, x->x.ir, nthreads)

function blocking(points::Vector{T}, by, nthreads=Sys.CPU_THREADS) where T <: SourcePoint
    sort!(points; by)

    # Put points into partitions such that no two partitions have the same finite-difference-grid-point.
    # In other words, if a new point is already in a partition, then it must go in that same partition.
    # This enables threading over partitions without race conditions. Note tha points that occupy the same
    # finie-difference-grid point will be adjacent in "points" due to the above sort.
    npartitions = min(nthreads, length(points))
    npartitions >= 1 || error("min(nthreads, length(points)) must be greater than zero, nthreads=$nthreads, length(points)=$(length(points))")

    nominal_points_per_partition,r = divrem(length(points), npartitions)
    if r > 0
        nominal_points_per_partition += 1
    end
    partitions = [T[] for _ = 1:npartitions]

    ipartition = 1
    point_counter = 1
    for ipoint in eachindex(points)
        if point_counter <= nominal_points_per_partition || (ipoint > 1 && by(points[ipoint]) == by(points[ipoint-1]))
            push!(partitions[ipartition], points[ipoint])
            point_counter += 1
        else
            ipartition += 1
            if ipartition > npartitions # this should never happen
                @warn "This algorithm is wrong, adding another partition"
                push!(partitions, T[])
            end
            push!(partitions[ipartition], points[ipoint])
            point_counter = 1
        end
    end

    empty_partition_indices = Int[]
    for (ipartition,partition) in enumerate(partitions)
        if isempty(partition)
            push!(empty_partition_indices, ipartition)
        end
    end
    deleteat!(partitions, empty_partition_indices)

    partitions
end

"""
  injectdata!(field, partitions, data, it, nthreads=Sys.CPU_THREADS)

Inject data from data[it] into field. `partitions` is computed using `WaveFD.source_blocking`.

The general 2D work-flow is:
```
nz,nx=size(field)
nthreads=20
partitions = WaveFD.source_blocking(nz,nx,iz,ix,ir,c,nthreads)
for it = 1:ntrec # time loop
    WaveFD.injectdata!(field, partitions, data, it, nthreads)
    ...
end
```
and the workflow for 3D is similar:
```
nz,ny,nx=size(field)
nthreads=20
partitions = WaveFD.source_blocking(nz,nx,iz,iy,ix,ir,c,nthreads)
for it = 1:ntrec # time loop
    WaveFD.injectdata!(field, partitions, data, it, nthreads)
    ...
end
```
"""
function injectdata!(field::Array{Float32}, partitions::Vector{<:Vector{SourcePoint32}}, data::Array, it::Int, nthreads=Sys.CPU_THREADS)
    npartitions = length(partitions)
    npoints_per_partition = [length(partition) for partition in partitions]
    nt = size(data, 1)
    @ccall libspacetime.injectdata_float(field::Ptr{Cfloat}, data::Ptr{Cfloat}, it::Csize_t, nt::Csize_t, partitions::Ptr{Ptr{SourcePoint32}}, npartitions::Csize_t, npoints_per_partition::Ptr{Clong}, nthreads::Csize_t)::Cvoid
end

function injectdata!(field::Array{Float64}, partitions::Vector{<:Vector{SourcePoint64}}, data::Array, it::Int, nthreads=Sys.CPU_THREADS)
    npartitions = length(partitions)
    npoints_per_partition = [length(partition) for partition in partitions]
    nt = size(data, 1)
    @ccall libspacetime.injectdata_double(field::Ptr{Cdouble}, data::Ptr{Cdouble}, it::Csize_t, nt::Csize_t, partitions::Ptr{Ptr{SourcePoint64}}, npartitions::Csize_t, npoints_per_partition::Ptr{Clong}, nthreads::Csize_t)::Cvoid
end

"""
  injectdata!(field, data, points[, nthreads=Sys.CPU_THREADS])
"""
function injectdata!(field::Array{T}, data::Array{T,2}, it::Integer, points, nthreads=Sys.CPU_THREADS) where {T}
    blocks = source_blocking(points, nthreads)
    injectdata!(field, blocks, data, it, nthreads)
end

"""
    extractdata!(data, field, it, points, nthreads=Sys.CPU_THREADS)
"""
function extractdata!(data::Array{Float32,2}, field::Array{Float32}, it::Integer, partitions::Vector{<:Vector{SourcePoint32}}, nthreads=Sys.CPU_THREADS)
    npartitions = length(partitions)
    npoints_per_partition = [length(partition) for partition in partitions]
    nt = size(data, 1)
    @ccall libspacetime.extractdata_float(data::Ptr{Cfloat}, field::Ptr{Cfloat}, partitions::Ptr{Ptr{SourcePoint32}}, npartitions::Csize_t, npoints_per_partition::Ptr{Clong}, it::Csize_t, nt::Csize_t, nthreads::Csize_t)::Cvoid
end

function extractdata!(data::Array{Float64,2}, field::Array{Float64}, it::Integer, partitions::Vector{<:Vector{SourcePoint64}}, nthreads=Sys.CPU_THREADS)
    npartitions = length(partitions)
    npoints_per_partition = [length(partition) for partition in partitions]
    nt = size(data, 1)
    @ccall libspacetime.extractdata_double(data::Ptr{Cdouble}, field::Ptr{Cdouble}, partitions::Ptr{Ptr{SourcePoint64}}, npartitions::Csize_t, npoints_per_partition::Ptr{Clong}, it::Csize_t, nt::Csize_t, nthreads::Csize_t)::Cvoid
end

function extractdata!(data::Array{T,2}, field::Array{T}, it, points::Vector{<:SourcePoint}, nthreads=Sys.CPU_THREADS) where {T}
    blocks = receiver_blocking(points, nthreads)
    extractdata!(data, field, it, blocks, nthreads)
end

#
# taper methods
#
function taper_region(nz, nx, iz, ix, iradius, iflat; interior=false)
    region = falses(nz, nx)
    addz = interior ? iflat : iflat+iradius
    addx = interior ? iflat : iflat+iradius
    for is = 1:length(ix)
        ilbz,iubz = clamp(minimum(iz[is])-addz,1,nz),clamp(maximum(iz[is])+addz,1,nz)
        ilbx,iubx = clamp(minimum(ix[is])-addx,1,nx),clamp(maximum(ix[is])+addx,1,nx)
        region[ilbz:iubz,ilbx:iubx] = true
    end

    region
end
function taper_region(nz, ny, nx, iz, iy, ix, iradius, iflat; interior=false)
    region = falses(nz, ny, nx)
    addz = interior ? iflat : iflat+iradius
    addy = interior ? iflat : iflat+iradius
    addx = interior ? iflat : iflat+iradius
    @inbounds for is = 1:length(ix)
        ilbz,iubz = clamp(minimum(iz[is])-addz,1,nz),clamp(maximum(iz[is])+addz,1,nz)
        ilby,iuby = clamp(minimum(iy[is])-addy,1,ny),clamp(maximum(iy[is])+addy,1,ny)
        ilbx,iubx = clamp(minimum(ix[is])-addx,1,nx),clamp(maximum(ix[is])+addx,1,nx)
        region[ilbz:iubz,ilby:iuby,ilbx:iubx] = true
    end

    region
end

function taper_region(taper_int::AbstractArray{Bool,2}, taper_ext)
    region_tpr = SVector{2,Float32}[]
    nz, nx = size(taper_int)
    @inbounds for jx = 1:nx, jz = 1:nz
        if !taper_int[jz,jx] && taper_ext[jz,jx]
            push!(region_tpr, SVector{2,Float32}(Float32(jz),Float32(jx)))
        end
    end
    region_tpr
end
function taper_region(taper_int::AbstractArray{Bool,3}, taper_ext)
    region_tpr = SVector{3,Float32}[]
    nz, ny, nx = size(taper_int)
    @inbounds for jx = 1:nx, jy = 1:ny, jz = 1:nz
        if !taper_int[jz,jy,jx] && taper_ext[jz,jy,jx]
            push!(region_tpr, SVector{3,Float32}(Float32(jz),Float32(jy),Float32(jx)))
        end
    end
    region_tpr
end

function taper_edge(region::AbstractArray{Bool,2})
    nz,nx = size(region)
    edge = SVector{2,Float32}[]
    @inbounds for ix = 2:nx-1, iz = 2:nz-1
        if region[iz,ix]
            if !region[iz+1,ix]
                push!(edge, SVector{2,Float32}(iz, ix))
                continue
            end
            if !region[iz-1,ix]
                push!(edge, SVector{2,Float32}(iz, ix))
                continue
            end
            if !region[iz,ix+1]
                push!(edge, SVector{2,Float32}(iz, ix))
                continue
            end
            if !region[iz,ix-1]
                push!(edge, SVector{2,Float32}(iz, ix))
            end
        end
    end
    KDTree(edge)
end
function taper_edge(region::AbstractArray{Bool,3})
    nz,ny,nx = size(region)
    edge = SVector{3,Float32}[]
    @inbounds for ix = 2:nx-1, iy = 2:ny-1, iz = 2:nz-1
        if region[iz,iy,ix]
            if !region[iz+1,iy,ix]
                push!(edge, SVector{3,Float32}(iz, iy, ix))
                continue
            end
            if !region[iz-1,iy,ix]
                push!(edge, SVector{3,Float32}(iz, iy, ix))
                continue
            end
            if !region[iz,iy+1,ix]
                push!(edge, SVector{3,Float32}(iz, iy, ix))
                continue
            end
            if !region[iz,iy-1,ix]
                push!(edge, SVector{3,Float32}(iz, iy, ix))
                continue
            end
            if !region[iz,iy,ix+1]
                push!(edge, SVector{3,Float32}(iz, iy, ix))
                continue
            end
            if !region[iz,iy,ix-1]
                push!(edge, SVector{3,Float32}(iz, iy, ix))
            end
        end
    end
    KDTree(edge)
end

function taper_build(t::Type{T}, nz, nx, region_tpr, region_int_edge, region_ext_edge) where T
    tpr = zeros(T, nz, nx)
    dint = knn(region_int_edge, region_tpr, 1)[2]
    dext = knn(region_ext_edge, region_tpr, 1)[2]
    for (i,tpr_pt) in enumerate(region_tpr)
        jz = Int(tpr_pt[1])
        jx = Int(tpr_pt[2])
        tpr[jz,jx] = dint[i][1]/(dint[i][1]+dext[i][1])
    end
    tpr
end
function taper_build(t::Type{T}, nz, ny, nx, region_tpr, region_int_edge, region_ext_edge) where T
    tpr = zeros(T, nz, ny, nx)
    dint = knn(region_int_edge, region_tpr, 1)[2]
    dext = knn(region_ext_edge, region_tpr, 1)[2]
    for (i,tpr_pt) in enumerate(region_tpr)
        jz = Int(tpr_pt[1])
        jy = Int(tpr_pt[2])
        jx = Int(tpr_pt[3])
        tpr[jz,jy,jx] = dint[i][1]/(dint[i][1]+dext[i][1])
    end
    tpr
end

function taper_etatozero(η, iz, ix, iradius, iflat)
    T = eltype(η)
    nz,nx = size(η)

    region_int = taper_region(nz, nx, iz, ix, iradius, iflat, interior=true)
    region_ext = taper_region(nz, nx, iz, ix, iradius, iflat, interior=false)

    region_int_edge = taper_edge(region_int)
    region_ext_edge = taper_edge(region_ext)
    region_tpr = taper_region(region_int, region_ext)

    tpr = taper_build(T, nz, nx, region_tpr, region_int_edge, region_ext_edge)

    for jx = 1:nx, jz = 1:nz
        if region_int[jz,jx] == 1
            η[jz,jx] = 0.0
            continue
        end
        if region_ext[jz,jx] == 1
            η[jz,jx] *= tpr[jz,jx]
        end
    end
    nothing
end
function taper_etatozero(η, iz, iy, ix, iradius, iflat)
    T = eltype(η)
    nz,ny,nx = size(η)

    region_int = taper_region(nz, ny, nx, iz, iy, ix, iradius, iflat, interior=true)
    region_ext = taper_region(nz, ny, nx, iz, iy, ix, iradius, iflat, interior=false)

    region_int_edge = taper_edge(region_int)
    region_ext_edge = taper_edge(region_ext)
    region_tpr = taper_region(region_int, region_ext)

    tpr = taper_build(T, nz, ny, nx, region_tpr, region_int_edge, region_ext_edge)

    for jx = 1:nx, jy = 1:ny, jz = 1:nz
        if region_int[jz,jy,jx] == 1
            η[jz,jy,jx] = 0.0
            continue
        end
        if region_ext[jz,jy,jx] == 1
            η[jz,jy,jx] *= tpr[jz,jy,jx]
        end
    end
    nothing
end

function taper_epsilon2delta(δ, ϵ, iz, ix, iradius, iflat)
    T = eltype(δ)
    nz,nx = size(δ)

    region_int = taper_region(nz, nx, iz, ix, iradius, iflat, interior=true)
    region_ext = taper_region(nz, nx, iz, ix, iradius, iflat, interior=false)

    region_int_edge = taper_edge(region_int)
    region_ext_edge = taper_edge(region_ext)
    region_tpr = taper_region(region_int, region_ext)

    tpr = taper_build(T, nz, nx, region_tpr, region_int_edge, region_ext_edge)

    for jx = 1:nx, jz = 1:nz
        if region_int[jz,jx]
            δ[jz,jx] = ϵ[jz,jx]
            continue
        end
        if region_ext[jz,jx]
            δ[jz,jx] = (1-tpr[jz,jx])*ϵ[jz,jx] + tpr[jz,jx]*δ[jz,jx]
        end
    end
    nothing
end
function taper_epsilon2delta(δ, ϵ, iz, iy, ix, iradius, iflat)
    T = eltype(δ)
    nz,ny,nx = size(δ)

    region_int = taper_region(nz, ny, nx, iz, iy, ix, iradius, iflat, interior=true)
    region_ext = taper_region(nz, ny, nx, iz, iy, ix, iradius, iflat, interior=false)

    region_int_edge = taper_edge(region_int)
    region_ext_edge = taper_edge(region_ext)
    region_tpr = taper_region(region_int, region_ext)

    tpr = taper_build(T, nz, ny, nx, region_tpr, region_int_edge, region_ext_edge)

    for jx = 1:nx, jy = 1:ny, jz = 1:nz
        if region_int[jz,jy,jx]
            δ[jz,jy,jx] = ϵ[jz,jy,jx]
            continue
        end
        if region_ext[jz,jy,jx]
            δ[jz,jy,jx] = (1-tpr[jz,jy,jx])*ϵ[jz,jy,jx] + tpr[jz,jy,jx]*δ[jz,jy,jx]
        end
    end
    nothing
end
