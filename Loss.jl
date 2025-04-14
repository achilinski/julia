# Loss.jl
module LossFunctions

using ..SimpleAutoDiff
using Statistics

export binary_cross_entropy

# Binary Cross Entropy Loss
# y_pred: Variable{T} (output of sigmoid, probability)
# y_true: Real (0.0 or 1.0)
function binary_cross_entropy(y_pred::Variable{T}, y_true::Real; ϵ=1e-9) where T<:Real # Capture T
    # Clipping is removed - rely on log's internal stability

    # BCE = -[y * log(p) + (1 - y) * log(1 - p)]
    # Create constants using type T from y_pred
    one_val = ones(T, size(value(y_pred)))
    one_var = Variable(one_val, is_param=false)

    # Ensure y_true is converted to T and wrapped correctly
    y_true_val = fill(T(y_true), size(value(y_pred)))
    y_true_var = Variable(y_true_val, is_param=false)

    # Use type-aware log and pass epsilon as T
    # log function will handle stability internally
    log_ypred = log(y_pred; ϵ=T(ϵ))
    log_one_minus_ypred = log(one_var - y_pred; ϵ=T(ϵ))

    # Calculation should now be type-stable: Var{T} * Var{T}
    term1 = y_true_var * log_ypred
    term2 = (one_var - y_true_var) * log_one_minus_ypred
    loss = -(term1 + term2) # Should be Variable{T}


    # If multiple outputs (batch), average the loss
    if length(value(loss)) > 1
         num_samples = length(value(loss))
         # Divide by Variable of type T containing the number of samples
         # Assuming loss is (batch_size, 1), denominator should match shape or broadcast
         denominator_val = fill(T(num_samples), size(value(loss)))
         denominator_var = Variable(denominator_val, is_param=false)
         # Use sum(loss) which returns Variable{T} (scalar) and divide by scalar Var{T}
         total_loss_var = sum(loss) # This reduces to scalar Variable{T}
         scalar_denom_var = Variable(T(num_samples), is_param=false) # Scalar Variable{T}
         return total_loss_var / scalar_denom_var # Now Var{T} / Var{T}
    else
        return loss # Return the scalar loss Variable
    end
end

end # module LossFunctions