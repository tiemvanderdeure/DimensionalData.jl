"""
    restore_array(data::AbstractVector, indices::AbstractVector{<:Integer}, dims::Tuple, missingval)

Restore a dimensional array from its tabular representation.

# Arguments
- `data`: An `AbstractVector` containing the flat data to be written to a `DimArray`.
- `indices`: An `AbstractVector` containing the flat indices corresponding to each element in `data`.
- `dims`: The dimensions of the destination `DimArray`.
- `missingval`: The value to write for missing elements in `data`.

# Returns
An `Array` containing the ordered valued in `data` with the size specified by `dims`.
```
"""
function restore_array(data::AbstractVector, indices::AbstractVector{<:Integer}, dims::Tuple, missingval)
    # Allocate Destination Array
    dst_size = prod(map(length, dims))
    dst = Vector{eltype(data)}(undef, dst_size)
    dst[indices] .= data

    # Handle Missing Rows
    _missingval = _cast_missing(data, missingval)
    missing_rows = ones(Bool, dst_size)
    missing_rows[indices] .= false
    data = ifelse.(missing_rows, _missingval, dst)

    # Reshape Array
    return reshape(data, size(dims))
end

"""
    coords_to_indices(table, dims; [selector], [atol])

Return the flat index of each row in `table` based on its associated coordinates.
Dimension columns are determined from the name of each dimension in `dims`.
It is assumed that the source/destination array has the same dimension order as `dims`.

# Arguments
- `table`: A table representation of a dimensional array. 
- `dims`: A `Tuple` of `Dimension` corresponding to the source/destination array.
- `selector`: The selector type to use. This defaults to `Near()` for orderd, sampled dimensions
    and `At()` for all other dimensions. 
- `atol`: The absolute tolerance to use with `At()`. This defaults to `1e-6`.


# Example
```julia
julia> d = DimArray(rand(256, 256), (X, Y));

julia> t = DimTable(d);

julia> coords_to_indices(t, dims(d))
65536-element Vector{Int64}:
     1
     2
     ⋮
 65535
 65536
```
"""
function coords_to_indices(table, dims::Tuple; selector=nothing, atol = 1e-6)
    return _coords_to_indices(table, dims, selector, atol)
end

# Find the order of the table's rows according to the coordinate values 
_coords_to_indices(table, dims::Tuple, sel, atol) = 
    _coords_to_indices(_dim_cols(table, dims), dims, sel, atol)
function _coords_to_indices(coords::NamedTuple, dims::Tuple, sel, atol)
    ords = _coords_to_ords(coords, dims, sel, atol)
    indices = _ords_to_indices(ords, dims)
    return indices
end

"""
    guess_dims(table; kw...)
    guess_dims(table, dims; precision=6)

Guesses the dimensions of an array based on the provided tabular representation.

# Arguments
- `table`: The input data table, which could be a `DataFrame`, `DimTable`, or any other Tables.jl compatible data structure. 
The dimensions will be inferred from the corresponding coordinate collumns in the table.
- `dims`: One or more dimensions to be inferred. If no dimensions are specified, then `guess_dims` will default
to any available dimensions in the set `(:X, :Y, :Z, :Ti, :Band)`. Dimensions can be given as either a singular
value or as a `Pair` with both the dimensions and corresponding order. The order will be inferred from the data
when none is given. This should work for sorted coordinates, but will not be sufficient when the table's rows are
out of order.
  
# Keyword Arguments
- `precision`: Specifies the number of digits to use for guessing dimensions (default = `6`).

# Returns
A tuple containing the inferred dimensions from the table.

# Example
```julia
julia> using DimensionalData, DataFrames

julia> import DimensionalData: Lookups, guess_dims

julia> xdims = X(LinRange{Float64}(610000.0, 661180.0, 2560));

julia> ydims = Y(LinRange{Float64}(6.84142e6, 6.79024e6, 2560));

julia> bdims = Dim{:Band}([:B02, :B03, :B04]);

julia> d = DimArray(rand(UInt16, 2560, 2560, 3), (xdims, ydims, bdims));

julia> t = DataFrame(d);

julia> t_rand = Random.shuffle(t);

julia> dims(d)
↓ X    Sampled{Float64} LinRange{Float64}(610000.0, 661180.0, 2560) ForwardOrdered Regular Points,
→ Y    Sampled{Float64} LinRange{Float64}(6.84142e6, 6.79024e6, 2560) ReverseOrdered Regular Points,
↗ Band Categorical{Symbol} [:B02, :B03, :B04] ForwardOrdered

julia> guess_dims(t)
↓ X    Sampled{Float64} 610000.0:20.0:661180.0 ForwardOrdered Regular Points,
→ Y    Sampled{Float64} 6.84142e6:-20.0:6.79024e6 ReverseOrdered Regular Points,
↗ Band Categorical{Symbol} [:B02, :B03, :B04] ForwardOrdered

julia> guess_dims(t, X, Y, :Band)
↓ X    Sampled{Float64} 610000.0:20.0:661180.0 ForwardOrdered Regular Points,
→ Y    Sampled{Float64} 6.84142e6:-20.0:6.79024e6 ReverseOrdered Regular Points,
↗ Band Categorical{Symbol} [:B02, :B03, :B04] ForwardOrdered

julia> guess_dims(t_rand, X => ForwardOrdered, Y => ReverseOrdered, :Band => ForwardOrdered)
↓ X    Sampled{Float64} 610000.0:20.0:661180.0 ForwardOrdered Regular Points,
→ Y    Sampled{Float64} 6.84142e6:-20.0:6.79024e6 ReverseOrdered Regular Points,
↗ Band Categorical{Symbol} [:B02, :B03, :B04] ForwardOrdered
```
"""
guess_dims(table; kw...) = guess_dims(table, _dim_col_names(table); kw...)
function guess_dims(table, dims::Tuple; precision=6, kw...)
    map(dim -> _guess_dims(get_column(table, name(dim)), dim, precision), dims)
end

"""
    get_column(table, dim::Type{<:Dimension})
    get_column(table, dim::Dimension)
    get_column(table, dim::Symbol)

Retrieve the coordinate data stored in the column specified by `dim`.

# Arguments
- `table`: The input data table, which could be a `DataFrame`, `DimTable`, or any other Tables.jl compatible data structure. 
- `dim`: A single dimension to be retrieved, which may be a `Symbol`, a `Dimension`.
"""
get_column(table, x::Type{<:Dimension}) = Tables.getcolumn(table, name(x))
get_column(table, x::Dimension) = Tables.getcolumn(table, name(x))
get_column(table, x::Symbol) = Tables.getcolumn(table, x)

"""
    data_col_names(table, dims::Tuple)

Return the names of all columns that don't matched the dimensions given by `dims`.

# Arguments
- `table`: The input data table, which could be a `DataFrame`, `DimTable`, or any other Tables.jl compatible data structure. 
- `dims`: A `Tuple` of one or more `Dimensions`.
"""
function data_col_names(table, dims::Tuple)
    dim_cols = name(dims)
    return filter(x -> !(x in dim_cols), Tables.columnnames(table))
end

_guess_dims(coords::AbstractVector, dim::Type{<:Dimension}, args...) = _guess_dims(coords, name(dim), args...)
_guess_dims(coords::AbstractVector, dim::Pair, args...) = _guess_dims(coords, first(dim), last(dim), args...)
function _guess_dims(coords::AbstractVector, dim::Symbol, ::Type{T}, precision::Int) where {T <: Order}
    return _guess_dims(coords, dim, T(), precision)
end
function _guess_dims(coords::AbstractVector, dim::Symbol, precision::Int)
    dim_vals = _dim_vals(coords, dim, precision)
    return format(Dim{dim}(dim_vals))
end
function _guess_dims(coords::AbstractVector, dim::Type{<:Dimension}, precision::Int)
    dim_vals = _dim_vals(coords, dim, precision)
    return format(dim(dim_vals))
end
function _guess_dims(coords::AbstractVector, dim::Dimension, precision::Int) 
    newl = _guess_dims(coords, lookup(dim), precision)
    return format(rebuild(dim, newl))
end
function _guess_dims(coords::AbstractVector, l::Lookup, precision::Int)
    dim_vals = _dim_vals(coords, l, precision)
    return rebuild(l; data = dim_vals)
end
# lookup(dim) could just return a vector - then we keep those values
_guess_dims(coords::AbstractVector, l::AbstractVector, precision::Int) = l

# Extract coordinate columns from table
function _dim_cols(table, dims::Tuple)
    dim_cols = name(dims)
    return NamedTuple{dim_cols}(Tables.getcolumn(table, col) for col in dim_cols)
end

# Extract dimension column names from the given table
_dim_col_names(table) = filter(x -> x in Tables.columnnames(table), (:X,:Y,:Z,:Ti,:Band))
_dim_col_names(table, dims::Tuple) = map(col -> Tables.getcolumn(table, col), name(dims))

# Extract data columns from table
function _data_cols(table, dims::Tuple)
    data_cols = data_col_names(table, dims)
    return NamedTuple{Tuple(data_cols)}(Tables.getcolumn(table, col) for col in data_cols)
end

# Determine the ordinality of a set of coordinates
_coords_to_ords(coords::Tuple, dims::Tuple, sel, atol) = map(args -> _coords_to_ords(args..., sel, atol), zip(coords, dims))
_coords_to_ords(coords::NamedTuple, dims::Tuple, sel, atol) = _coords_to_ords(map(x -> coords[x], name(dims)), dims, sel, atol)
# implement some default selectors
_coords_to_ords(coords::AbstractVector, dim::Dimension{<:AbstractCategorical}, sel::Nothing, atol) = 
    _coords_to_ords(coords, dim, At(), atol)
function _coords_to_ords(coords::AbstractVector, dim::Dimension{<:AbstractSampled}, sel::Nothing, atol) 
    sel = if sampling(dim) isa Intervals
        Contains()
    elseif isordered(dim) 
        Near()
    else 
        At()
    end
    _coords_to_ords(coords, dim, sel, atol)
end
_coords_to_ords(coords::AbstractVector, dim::Dimension, sel::Nothing) = 
    _coords_to_ords(coords, dim, Near(), atol)

# get indices of the coordinates
function _coords_to_ords(
    coords::AbstractVector, 
    dim::Dimension, 
    sel::Selector,
    atol)
    return map(c -> selectindices(dim, rebuild(sel, c)), coords)
end
# get indices of the coordinates
function _coords_to_ords(
    coords::AbstractVector, 
    dim::Dimension, 
    sel::At,
    atol)
    return map(c -> selectindices(dim, rebuild(sel; val = c, atol)), coords)
end

# Round coordinate ordinality to the appropriate integer given the specified locus
_round_ords(ords::AbstractVector{<:Real}, ::DimensionalData.Start) = floor.(Int, ords)
_round_ords(ords::AbstractVector{<:Real}, ::DimensionalData.Center) = round.(Int, ords)
_round_ords(ords::AbstractVector{<:Real}, ::DimensionalData.End) = ceil.(Int, ords)

# Extract dimension value from the given vector of coordinates
function _dim_vals(coords::AbstractVector, dim, precision::Int)
    vals = _unique_vals(coords, precision)
    return _maybe_as_range(vals, precision)
end
function _dim_vals(coords::AbstractVector, l::Lookup, precision::Int)
    val(l) isa AutoValues || return val(l) # do we want to have some kind of check that the values match?
    vals = _unique_vals(coords, precision)
    _maybe_order!(vals, order(l))
    return _maybe_as_range(vals, precision)
end
_dim_vals(coords::AbstractVector, l::AbstractVector, precision::Int) = l # same comment as above?

_maybe_order!(A::AbstractVector, ::Order) = A
_maybe_order!(A::AbstractVector, ::ForwardOrdered) = sort!(A)
_maybe_order!(A::AbstractVector, ::ReverseOrdered) = sort!(A, rev=true)

# Extract all unique coordinates from the given vector
_unique_vals(coords::AbstractVector, ::Int) = unique(coords)
_unique_vals(coords::AbstractVector{<:Real}, precision::Int) = round.(coords, digits=precision) |> unique
_unique_vals(coords::AbstractVector{<:Integer}, ::Int) = unique(coords)

# Estimate the span between consecutive coordinates
_maybe_as_range(A::AbstractVector, precision) = A
function _maybe_as_range(A::AbstractVector{<:Real}, precision::Int)
    A_r = range(first(A), last(A), length(A))
    atol = 10.0^(-precision)
    return all(i -> isapprox(A_r[i], A[i]; atol), eachindex(A)) ? A_r : A
end
function _maybe_as_range(A::AbstractVector{<:Integer}, precision::Int)
    idx1, idxrest = Iterators.peel(eachindex(A))
    step = A[idx1+1] - A[idx1]
    for idx in idxrest
        A[idx] - A[idx-1] == step || return A
    end
    return first(A):step:last(A)
end
function _maybe_as_range(A::AbstractVector{<:Dates.AbstractTime}, precision::Int)
    steps = (@view A[2:end]) .- (@view A[1:end-1])
    span = argmin(abs, steps)
    isregular = all(isinteger, round.(steps ./ span, digits=precision))
    return isregular ? range(first(A), last(A), length(A)) : A
end

# Determine the index from a tuple of coordinate orders
function _ords_to_indices(ords, dims)
    stride = 1
    indices = ones(Int, length(ords[1]))
    for (ord, dim) in zip(ords, dims)
        indices .+= (ord .- 1) .* stride
        stride *= length(dim)
    end
    return indices
end

_cast_missing(::AbstractArray, missingval::Missing) = missing
function _cast_missing(::AbstractArray{T}, missingval) where {T}
    try
        return convert(T, missingval)
    catch e
        return missingval
    end
end