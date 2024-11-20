srcillum_helper(field_ginsu::Array{T}, time_mask::T, nthreads) where {T<:Float32} =
    ccall((:srcillum_helper_float,libillumination), Cvoid,
        (Ptr{Float32},Cfloat,Csize_t,Cint), field_ginsu, time_mask, length(field_ginsu), nthreads)

srcillum_helper(field_ginsu::Array{T}, time_mask::T, nthreads) where {T<:Float64} =
    ccall((:srcillum_helper_double,libillumination), Cvoid,
        (Ptr{Float64},Cdouble,Csize_t,Cint), field_ginsu, time_mask, length(field_ginsu), nthreads)

illum_accumulate(field_ginsu_accum::Array{T}, field_ginsu::Array{T}, time_mask::T, nthreads) where {T<:Float32} =
    ccall((:illum_accumulate_float,libillumination), Cvoid,
        (Ptr{Float32},Ptr{Float32},Cfloat,Csize_t,Cint), field_ginsu_accum, field_ginsu, time_mask, length(field_ginsu), nthreads)

illum_accumulate(field_ginsu_accum::Array{T}, field_ginsu::Array{T}, time_mask::T, nthreads) where {T<:Float64} =
    ccall((:illum_accumulate_double,libillumination), Cvoid,
        (Ptr{Float64},Ptr{Float64},Cdouble,Csize_t,Cint), field_ginsu_accum, field_ginsu, time_mask, length(field_ginsu), nthreads)
