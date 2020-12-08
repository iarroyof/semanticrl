using PyCall
using ScikitLearn
using CSV
using DataFrames

@sk_import feature_extraction.text: TfidfVectorizer


input_tsv_triplets = "/almac/ignacio/semanticrl/data/dis_train.txt.oie"


mutable Struct SetHashedDict
    mem::Dict{Tuple{Set{String}, Set{String}}, Float64}
    rv_mem::Dict{Set{String}, Float64}
end


function build_tokenizer(ngramr::Tuple)
    text_preprocessor = TfidfVectorizer(analyzer="char_wb", ngram_range=ngramr)
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

function build_set_vocabs(x_set::Set{String}, y_set::Set{String}, z_set::Set{String})
    omega_x = Set{String}[]
    omega_y = Set{String}[]
    omega_z = Set{String}[]

    for x in x_set
        if x in omega_x:
            continue
        else
            append!(omega_x, x)
        end
    end

    for y in y_set
        if y in omega_y:
            continue
        else
            append!(omega_y, y)
        end
    end

    for z in z_set
        if z in omega_z:
            continue
        else
            append!(omega_z, z)
        end
    end

    return (omega_x, omega_y, omega_z)
end


function intersect_lens(Oa, B)
    # Generate cartesian product (Oa x B) for all \omega in Oa and b in B
    cart_prod = [repeat(Oa, inner=[size(B, 1)]) repeat(B, outer=[size(Oa, 1)])]

    return zip(cart_prod, length.(intersect.(cart_prod)))
end


function compute_intersects(omega_x, omega_y, omega_x, X, Y, Z)

    ints_x = intersect_lens(omega_x, X)
    ints_y = intersect_lens(omega_y, Y)
    ints_z = intersect_lens(omega_z, Z)

    ints_xy = intersect_lens(omega_x, Y)
    ints_yz = intersect_lens(omega_y, Z)
    ints_zx = intersect_lens(omega_z, X)

    return (ints_x, ints_y, ints_z, ints_xy, ints_yz, ints_zx)
end


function compute_proba(omega_a::Array, Ua::Base.Iterators.Zip, omega_b::Array, Ub::Base.Iterators.Zip, mem::SetHashedDict)
    P_AB = Dict{Set, Float64}
    for a in omega_a
        aint_lens = [b for (a_, b) in zip(Ua[:, 1], Ua[:, 2]) if a_ == a]
        for b in omega_b
            bint_lens = [b for (a_, b) in zip(Ub[:, 1], Ub[:, 2]) if a_ == b]
        end
        P_AB[a] = aint_lens * bint_lens
    end

    return P_AB


function compute_probas()        
    mem_xy = SetHashedDict()
    mem_yz = SetHashedDict()
    mem_zx = SetHashedDict()

    for x in omega_x
        for 



function main()
    X_set, Y_set, Z_set = build_set_RVs(input_tsv_triplets, (1, 4));
    omega_x, omega_y, omega_z = build_set_vocabs(X_set, Y_set, Z_set);
