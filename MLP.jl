# MLP.jl
module MLPDefinition

using ..SimpleAutoDiff # Use the custom AD
using Random, LinearAlgebra

# Export Dense, activation functions, MLPModel structure, forward, and get_params
export Dense, relu, sigmoid, MLPModel, forward, get_params

# --- Activation Functions ---

# Ensure max comparison uses the same type T as the input Variable
relu(x::Variable{T}) where T<:Real = max(x, T(0.0))

# Sigmoid function, capturing type T for constants
function sigmoid(x::Variable{T}; ϵ=1e-8) where T<:Real
    # Numerically stable sigmoid: σ(x) = 1 / (1 + exp(-x))

    # Create 'one' with the same type T as x
    one_val = ones(T, size(value(x)))
    one_var = Variable(one_val, is_param=false)

    # Ensure exp(-x) is also Variable{T} (unary minus and exp should preserve T now)
    exp_neg_x = exp(-x)

    # Calculation should now be type-stable: Var{T} / (Var{T} + Var{T})
    # Add a small epsilon to the denominator for division stability, also of type T
    denominator = one_var + exp_neg_x + Variable(fill(Base.eps(T), size(value(x))), is_param=false)

    return one_var / denominator
end


# --- Dense Layer ---
struct Dense
    W::Variable # Weights should be Variable{T}
    b::Variable # Biases should be Variable{T}
    activation::Function

    # Constructor ensuring weights/biases match specified dtype (defaults to Float32)
    function Dense(input_size::Int, output_size::Int, activation::Function=identity; dtype::Type{<:Real}=Float32)
        # Xavier/Glorot initialization respecting dtype
        limit = sqrt(dtype(6.0) / (dtype(input_size) + dtype(output_size)))
        W_val = rand(dtype, input_size, output_size) .* dtype(2.0) .* limit .- limit
        b_val = zeros(dtype, 1, output_size) # Bias as row vector

        W = Variable(W_val, is_param=true) # Creates Variable{dtype}
        b = Variable(b_val, is_param=true) # Creates Variable{dtype}
        new(W, b, activation)
    end
end

# Forward pass for Dense layer
# Input x is Variable{T}, layer W and b should also be Variable{T}
function forward(layer::Dense, x::Variable{T}) where T
    # matmul(Var{T}, Var{T}) -> Var{T}
    # Var{T} + Var{T} -> Var{T} (bias addition involves broadcasting)
    # activation(Var{T}) -> Var{T}
    output = matmul(x, layer.W) + layer.b
    return layer.activation(output)
end

# Get parameters (W, b) from the layer
function get_params(layer::Dense)
    return [layer.W, layer.b]
end


# --- MLP Model ---
struct MLPModel
    layers::Vector{Any} # Can hold Dense layers or other types
    parameters::Vector{Variable} # Should collect all Variable{T} params

    # Constructor collecting layers and their parameters
    function MLPModel(layers...)
        model_layers = [l for l in layers]
        params = Variable[] # Initialize as Vector{Variable}
        for layer in model_layers
            # Check if the layer has a specific get_params method defined
            if hasmethod(get_params, (typeof(layer),))
                # Use append! to add elements from the layer's params vector
                append!(params, get_params(layer))
            end
        end
        new(model_layers, params)
    end
end

# Forward pass for the whole MLP
function forward(model::MLPModel, x::Variable{T}) where T # Pass T through
    out = x
    for layer in model.layers
        # Pass Variable{T} through each layer's forward pass
        out = forward(layer, out)
    end
    return out # Final output should be Variable{T}
end

# Allow calling the model like a function, maintaining type T
(model::MLPModel)(x::Variable{T}) where T = forward(model, x)

# Get all parameters collected during construction
function get_params(model::MLPModel)
    return model.parameters
end


end # module MLPDefinition