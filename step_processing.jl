using ScikitLearn
using DataFrames
using IterTools
using Statistics
using StatsBase
using Printf
using PyCall
using CSV
using JLD
using HDF5

@sk_import feature_extraction.text: TfidfVectorizer


#=
TODO: 1. To add density function (Gaussian and Exponential)
TODO: 2. PMF probabilities are the expected value over the sample space
TODO: 3. Verify wheter this programs is faster than python version (or how to 
         optimize, cartesian products are really needed?): 778.174950 seconds 
         (24.45 G allocations: 733.674 GiB, 66.92% gc time) (13 min, 92 steps,
         320 samples)
TODO: 4. To add Entropy, Conditional Entropy, Mutual Information and Divergence
TODO: 5. Probability surfaces inducing subordered set-valued outcomes
=#

function build_tokenizer(ngramr::Tuple)
    text_preprocessor = TfidfVectorizer(
        analyzer="char_wb", ngram_range=ngramr)
    tokenizer = text_preprocessor.build_analyzer()
    return tokenizer
end


function build_set_RVs(df::DataFrame, ngramer::Tuple)

    tokenizer = build_tokenizer(ngramer)

    X = Set.(tokenizer.(df.Column2))
    Y = Set.(tokenizer.(df.Column3))
    Z = Set.(tokenizer.(df.Column4))

    return (X, Y, Z)
end


function build_set_vocabs(x_set::Array{Set{String}, 1},
        y_set::Array{Set{String}, 1}, z_set::Array{Set{String}, 1})
    omega_x = []
    omega_y = []
    omega_z = []

    for x in x_set
        if x in omega_x
            continue
        else
            push!(omega_x, x)
        end
    end

    for y in y_set
        if y in omega_y
            continue
        else
            push!(omega_y, y)
        end
    end

    for z in z_set
        if z in omega_z
            continue
        else
            push!(omega_z, z)
        end
    end

    return (omega_x, omega_y, omega_z)
end


function intersect_lens(Oa, B)
    # Generate cartesian product (Oa x B) for all \omega in Oa and b in B
    cart_prod = product(Oa, B)
    lengths = length.(broadcast(a -> intersect(a[1], a[2])[1],
                                eachrow(collect(cart_prod))))
    return zip(collect(cart_prod), lengths)
end


function compute_intersects(omega_x, omega_y, omega_z, X, Y, Z)

    ints_x = intersect_lens(omega_x, X)
    ints_y = intersect_lens(omega_y, Y)
    ints_z = intersect_lens(omega_z, Z)

    ints_xy = intersect_lens(omega_x, Y)
    ints_yz = intersect_lens(omega_y, Z)
    ints_zx = intersect_lens(omega_z, X)

    return (ints_x, ints_y, ints_z, ints_xy, ints_yz, ints_zx)
end


function compute_proba(omega_a, ints_a, omega_b, ints_b)
    set_hash = []
    b_part = []
    
    for b in omega_b
        partition = 0
        bint_lens = [c for ((a_, b_), c) in ints_b if b == a_]

        for a in omega_a
            aint_lens = [c for ((a_, b_), c) in ints_a if a == a_]
            a_dot_b = aint_lens'bint_lens
            push!(set_hash, ((a, b), a_dot_b))
            partition += a_dot_b
        end
        push!(b_part, (b, partition))
    end
    set_hash = Dict(set_hash)
    b_part = Dict(b_part)
    f_AB = []
    for b in omega_b
        for a in omega_a
            push!(f_AB, ((a, b), set_hash[(a, b)]/b_part[b]))
        end
    end
    f_AB = Dict(f_AB)

    P_AB = []
    for a in omega_a
        push!(P_AB, (a, mean(f_AB[(a, b)] for b in omega_b)))
    end

    return (f_AB, Dict(P_AB))
end


function compute_probas(ixx, iyy, izz, ixy, iyz, izx, ohm_x, ohm_y, ohm_z)
    f_xx, P_XX = compute_proba(ohm_x, ixx, ohm_x, ixx)
    f_yy, P_YY = compute_proba(ohm_y, iyy, ohm_y, iyy)
    f_zz, P_ZZ = compute_proba(ohm_z, izz, ohm_z, izz)

    f_ygx, P_YgX = compute_proba(ohm_y, iyy, ohm_x, ixx)
    f_zgy, P_ZgY = compute_proba(ohm_z, izz, ohm_y, iyy)
    f_zgx, P_ZgX = compute_proba(ohm_z, izz, ohm_x, ixx)

    f_xgy, P_XgY = compute_proba(ohm_x, ixx, ohm_y, iyy)
    f_ygz, P_YgZ = compute_proba(ohm_y, iyy, ohm_z, izz)
    f_xgz, P_XgZ = compute_proba(ohm_x, ixx, ohm_z, izz)

    return Dict("f_X" => f_xx, "f_Y" => f_yy, "f_Z" => f_zz,
                "f_Y|X" => f_ygx, "f_Z|Y" => f_zgy, "f_Z|X" => f_zgx,
                "P_X" => P_XX, "P_Y" => P_YY, "P_Z" => P_ZZ,
                "P_Y|X" => P_YgX, "P_Z|Y" => P_ZgY, "P_Z|X" => P_ZgX,
                "f_X|Y" => f_xgy, "f_Y|Z" => f_ygz, "f_X|Z" => f_xgz,
                "P_X|Y" => P_XgY, "P_Y|Z" => P_YgZ, "P_X|Z" => P_XgZ)
end


function entropy2(P)
    return entropy(values(P), 2.0)
end


function cond_entropy(P_YgX, P_X, omega_y, omega_x)
    H_Ygx = []
    for x in omega_x
        push!(H_Ygx, P_X[x] * entropy2(P_YgX[(y, x)] for y in omega_y))
    end

    H_YgX = sum(H_Ygx)

    return H_YgX
end


function mutual_inf(H_Y, H_YgX)

    I_YX = H_Y - H_YgX
    
    return I_YX
end


function compute_it(D, omega_x, omega_y, omega_z)
    IT = Dict(
    "H_X" => entropy2(D["P_X"]),
    "H_Y" => entropy2(D["P_Y"]),
    "H_Z" => entropy2(D["P_Z"]),

    "H_YgX" => cond_entropy(D["f_Y|X"], D["P_X"], omega_y, omega_x),
    "H_ZgY" => cond_entropy(D["f_Z|Y"], D["P_Y"], omega_z, omega_y),
    "H_ZgX" => cond_entropy(D["f_Z|X"], D["P_X"], omega_z, omega_x),

    "H_XgY" => cond_entropy(D["f_X|Y"], D["P_Y"], omega_x, omega_y),
    "H_YgZ" => cond_entropy(D["f_Y|Z"], D["P_Z"], omega_y, omega_z),
    "H_XgZ" => cond_entropy(D["f_X|Z"], D["P_Z"], omega_x, omega_z))

    IT["I_YX"] = mutual_inf(IT["H_Y"], IT["H_YgX"])
    IT["I_ZY"] = mutual_inf(IT["H_Z"], IT["H_ZgY"])
    IT["I_ZX"] = mutual_inf(IT["H_Z"], IT["H_ZgX"])

    IT["I_XY"] = mutual_inf(IT["H_X"], IT["H_XgY"])
    IT["I_YZ"] = mutual_inf(IT["H_Y"], IT["H_YgZ"])
    IT["I_XZ"] = mutual_inf(IT["H_X"], IT["H_XgZ"])

    return IT
end


function process_step(X_set, Y_set, Z_set)
    Omega_x, Omega_y, Omega_z = build_set_vocabs(X_set, Y_set, Z_set)
    xx, yy, zz, xy, yz, zx = compute_intersects(
                            Omega_x, Omega_y, Omega_z, X_set, Y_set, Z_set)
    dists = compute_probas(xx, yy, zz, xy, yz, zx, Omega_x, Omega_y, Omega_z)
    inf_theo = compute_it(dists, Omega_x, Omega_y, Omega_z)

    return inf_theo
end


function main()
    input_tsv_triplets = "data/dis_train.txt.oie"
    output_csv_it = "results/train_results_julia_320.csv"
    step_size = 320
    inputs = []
    @time begin
    for rows in Iterators.partition(
            CSV.Rows(input_tsv_triplets, delim='\t', header=false), step_size)
        df = DataFrame(rows)
        push!(inputs, build_set_RVs(df, (1, 4)))
    end
    println("Chunks created... ")

    n_steps = length(inputs)
    results = Array{Dict{String, Float64}, 1}(undef, n_steps)
    println("Ready to process a total of " * string(n_steps) *
                        "steps of " * string(step_size) * " samples each...")
    Threads.@threads for (i, (X_set, Y_set, Z_set)) in collect(
                                                            enumerate(inputs))
        @time begin
        it = process_step(X_set, Y_set, Z_set)
        results[i] = it
        println("Step " * string(i))
        end
    end
    println("All finished... Saving output CSV")
    out_df = vcat(DataFrame.(results)...)
    CSV.write(output_csv_it, out_df)
    end
end

main()

