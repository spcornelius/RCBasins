# utility function that frustratingly doesn't already exist
# in Configurations.jl; load a configuration from a YAML file
function from_yaml(::Type{T}, filename::String) where {T}
    d = YAML.load_file(filename; dicttype=Dict{String, Any})
    from_dict(T, d)
end

# convert a YAML list to a tuple
function Configurations.from_dict(::Type{OptionType}, of::OptionField, 
                                  ::Type{T}, x::AbstractVector) where {OptionType, T<:Tuple}
    return tuple(x...)
end

# interpolate strings arguments using using Mustache
function Configurations.from_dict(::Type{OptionType}, of::OptionField, 
    ::Type{String}, x::String) where {OptionType}
    # do substitutions for locals within NextGenRC, then in Main
    tmp = render(x, Config)
    return render(tmp, Main)
end