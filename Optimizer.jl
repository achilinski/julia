# Optimizer.jl
module Optimizers

using ..SimpleAutoDiff

export SGD, update!

# Make SGD parametric on the type T of the parameters/learning rate
mutable struct SGD{T<:Real}
    lr::T
    params::Vector{<:Variable{T}} # Ensure params are also of type T

    # Constructor enforces type consistency
    function SGD(lr::T, params::Vector{<:Variable{T}}) where {T<:Real}
        new{T}(lr, params)
    end
end

# Update function now uses the type T from the optimizer
function update!(opt::SGD{T}) where {T<:Real}
    for p in opt.params
        if p.gradient !== nothing && p.is_param
            # Gradient should already be accumulated as type T by accumulate_gradient!
            grad_val = grad(p) # This is Union{T, AbstractArray{T}, Nothing}

            # Check for NaN/Inf gradients before update
            # Handle both scalar and array gradients
            if (isa(grad_val, Real) && (isnan(grad_val) || isinf(grad_val))) || (isa(grad_val, AbstractArray) && (any(isnan, grad_val) || any(isinf, grad_val)))
                 println("Warning: NaN or Inf gradient detected for parameter shape $(size(p.value)). Skipping update.")
                 continue
            end

            # Update parameter value (in-place modification)
            # p.value is Array{T} or T
            # opt.lr is T
            # grad_val should be Array{T} or T
            # Operation: T .-= T .* T -> maintains type T
            p.value .-= opt.lr .* grad_val

        elseif p.is_param && p.gradient === nothing
             # This might be okay if a parameter wasn't used in a specific forward pass
             # println("Debug: Parameter $(size(p.value)) has no gradient during update.")
        end
    end
end

end # module Optimizers