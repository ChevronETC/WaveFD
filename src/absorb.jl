function setup_q_profile_2D_serial!(w_inv_q::Array{T,2}, freesurface::Bool, nsponge::Int64, 
    dt::T, q_freq::T, q_min::T, q_interior::T) where T<:Float32
    # check for unphysical q pairing
    if (q_min < eps(T)) && (q_interior < eps(T))
        @info "q_min & q_interior set to less than machine epsilon; setting wOverQ to zero"
        w_inv_q .= zero(T)
        return
    end
    nz,nx = size(w_inv_q)
    lqmin = log(q_min)
    lqmax = log(q_interior)
    qprof = zeros(Float32, nsponge)

    for ksponge = 1:nsponge
        dk = (ksponge - 1) / (nsponge - 1)
        lq = lqmin + dk * (lqmax - lqmin)
        qprof[ksponge] = T(exp(lq))
    end

    for kz = 1:nz
        ksz = (freesurface) ? (nz - 1 - (kz - 1)) : min((kz - 1), (nz - 1 - (kz - 1)))
        for kx = 1:nx
            ksx = min((kx - 1),   (nx - 1 - (kx - 1)))
            ksponge = min(ksx, ksz) + 1 # add one to get back to julia indexing
            @inbounds w_inv_q[kz,kx] = (dt * 2.0 * π * q_freq / q_interior)
            if ksponge ∈ 1:nsponge
                @inbounds w_inv_q[kz,kx] = dt * 2.0 * π * q_freq / qprof[ksponge]
            end
        end
    end
end


function make_qprof(nsponge::Int64, q_min::T, q_interior::T) where T<:Float32
    lqmin = log(q_min)
    lqmax = log(q_interior)
    qprof = zeros(Float32, nsponge)

    @threads for ksponge = 1:nsponge
        dk = (ksponge - 1) / (nsponge - 1)
        lq = lqmin + dk * (lqmax - lqmin)
        @inbounds qprof[ksponge] = exp(lq)
    end

    return qprof
end

function setup_q_profile_2D_threaded!(w_inv_q::Array{T,2}, freesurface, nsponge::Int64, 
        dt::T, q_freq::T, q_min::T, q_interior::T) where T<:Float32
    # check for unphysical q pairing
    if (q_min < eps(T)) && (q_interior < eps(T))
        @info "q_min & q_interior set to less than machine epsilon; setting wOverQ to zero"
        w_inv_q .= zero(T)
        return
    end
    
    function compute_w_inv_q_2D!(w_inv_q::Array{T,2}, qprof::Array{T,1}, freesurface::Bool, nsponge::Int64, q_freq::T, q_interior::T) where T<:Float32
        nz,nx = size(w_inv_q)

        @threads for kz = 1:nz
            ksz = (freesurface) ? (nz - 1 - (kz - 1)) : min((kz - 1), (nz - 1 - (kz - 1)))
            for kx = 1:nx
                ksx = min((kx - 1),   (nx - 1 - (kx - 1)))
                ksponge = min(ksx, ksz) + 1 # add one to get back to julia indexing
                @inbounds w_inv_q[kz,kx] = dt * 2.0 * π * q_freq / q_interior
                if ksponge ∈ 1:nsponge
                    @inbounds w_inv_q[kz,kx] = dt * 2.0 * π * q_freq / qprof[ksponge]
                end
            end
        end
    end

    qprof = make_qprof(nsponge, q_min, q_interior)
    compute_w_inv_q_2D!(w_inv_q, qprof, freesurface, nsponge, q_freq, q_interior)
end

function setup_q_profile_3D_serial!(w_inv_q::Array{T,3}, freesurface::Bool, nsponge::Int64, 
    dt::T, q_freq::T, q_min::T, q_interior::T) where T<:Float32
    # check for unphysical q pairing
    if (q_min < eps(T)) && (q_interior < eps(T))
        @info "q_min & q_interior set to less than machine epsilon; setting wOverQ to zero"
        w_inv_q .= zero(T)
        return
    end
    
    nz,ny,nx = size(w_inv_q)
    lqmin = log(q_min)
    lqmax = log(q_interior)
    qprof = zeros(Float32, nsponge)

    for ksponge = 1:nsponge
        dk = (ksponge - 1) / (nsponge - 1)
        lq = lqmin + dk * (lqmax - lqmin)
        qprof[ksponge] = T(exp(lq))
    end

    for kz = 1:nz
        ksz = (freesurface) ? (nz - 1 - (kz - 1)) : min((kz - 1), (nz - 1 - (kz - 1)))
        
        for ky = 1:ny
            ksy = min((ky - 1),   (ny - 1 - (ky - 1)))
            for kx = 1:nx
                ksx = min((kx - 1),   (nx - 1 - (kx - 1)))
                ksponge = min(ksx,min(ksy, ksz)) + 1 # add one to get back to julia indexing
                @inbounds w_inv_q[kz,ky,kx] = (dt * 2.0 * π * q_freq / q_interior)
                if ksponge ∈ 1:nsponge
                    @inbounds w_inv_q[kz,ky,kx] = dt * 2.0 * π * q_freq / qprof[ksponge]
                end
            end
        end
    end
end

function setup_q_profile_3D_threaded!(w_inv_q::Array{T,3}, freesurface, nsponge::Int64, 
    dt::T, q_freq::T, q_min::T, q_interior::T) where T<:Float32

    # check for unphysical q pairing
    if (q_min < eps(T)) && (q_interior < eps(T))
        @info "q_min & q_interior set to less than machine epsilon; setting wOverQ to zero"
        w_inv_q .= zero(T)
        return
    end

    function compute_w_inv_q_3D!(w_inv_q::Array{T,3}, qprof::Array{T,1}, freesurface::Bool, nsponge::Int64, q_freq::T, q_interior::T) where T<:Float32
        nz,ny,nx = size(w_inv_q)

        @threads for kz = 1:nz
            ksz = (freesurface) ? (nz - 1 - (kz - 1)) : min((kz - 1), (nz - 1 - (kz - 1)))
            
            for ky = 1:ny
                ksy = min((ky - 1),   (ny - 1 - (ky - 1)))
                for kx = 1:nx
                    ksx = min((kx - 1),   (nx - 1 - (kx - 1)))
                    ksponge = min(ksx,min(ksy, ksz)) + 1 # add one to get back to julia indexing
                    @inbounds w_inv_q[kz,ky,kx] = dt * 2.0 * π * q_freq / q_interior
                    if ksponge ∈ 1:nsponge
                        @inbounds w_inv_q[kz,ky,kx] = dt * 2.0 * π * q_freq / qprof[ksponge]
                    end
                end
            end
        end
    end

    qprof = make_qprof(nsponge, q_min, q_interior)
    compute_w_inv_q_3D!(w_inv_q, qprof, freesurface, nsponge, q_freq, q_interior)
end
