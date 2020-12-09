using PyCall
using ScikitLearn
using CSV
using DataFrames
using IterTools

@sk_import feature_extraction.text: TfidfVectorizer


input_tsv_triplets = "/almac/ignacio/semanticrl/data/dis_train.txt.oie"
    

mutable struct SetHashedDict
       set_hash::Dict{Tuple{Set{String}, Set{String}}, Float64}
       rv_mem::Dict{Set{String}, Float64}
end


function build_tokenizer(ngramr::Tuple)
    text_preprocessor = TfidfVectorizer(
        analyzer="char_wb", ngram_range=ngramr)
    tokenizer = text_preprocessor.build_analyzer()
    return tokenizer
end


function build_set_RVs(input_tsv::String, ngramer::Tuple)
    df = DataFrame(CSV.File(input_tsv, delim='\t', header=false))

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
    lengths = length.(broadcast(a -> intersect(a[1], a[2]), eachrow(cart_prod)))
    return zip(permutedims(Tuple.(eachrow(cart_prod))), permutedims(lengths))
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
    rv_mem = []
    
    for b in omega_b
        partition = 0
        bint_lens = [c for ((a_, b_), c) in ints_b if b == a_]

        for a in omega_a
            aint_lens = [c for ((a_, b_), c) in ints_a if a == a_]
            a_dot_b = aint_lens'bint_lens
            push!(set_hash, ((a, b), a_dot_b))
            partition += produ
        end
    push!(rv_mem, (b, partition))
    end
    set_hash = Dict(set_hash)
    rv_mem = Dict(rv_mem)
    P_AB = Dict([((a, b), set_hash[(a, b)]/rv_mem[b]) for (a, b) in product(omega_a, omega_b)])

    return P_AB
end


function compute_probas()        
    mem_xy = SetHashedDict()
    mem_yz = SetHashedDict()
    mem_zx = SetHashedDict()

    for x in omega_x
        for 



function main()
    X_set, Y_set, Z_set = build_set_RVs(input_tsv_triplets, (1, 4));
    omega_x, omega_y, omega_z = build_set_vocabs(X_set, Y_set, Z_set);
    ints_x, ints_y, ints_z, ints_xy, ints_yz, ints_zx = compute_intersects(omega_x, omega_y, omega_x, X_set, Y_set, Z_set)
    
    
