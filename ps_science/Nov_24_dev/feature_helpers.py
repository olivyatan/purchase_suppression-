import pyspark.sql.functions as F
from pyspark.sql.types import FloatType, ArrayType, DoubleType 
import math
import re
import numpy as np
from collections import Counter



@F.udf(returnType=DoubleType())
def entropy(proportions):
    if proportions is None or len(proportions) == 0:
        return -1  
    entropy_value = -sum([p * math.log2(p) for p in proportions if p > 0])
    return round(abs(entropy_value), 4)


def clean_text(txt):
    return re.sub(r'\W+', ' ', txt)

def tokenize_text(txt):
    return " ".join(txt).lower().encode("utf-8").split()

def to_vec(tokens):
    v = Counter(tokens)
    v_norm = math.sqrt(math.fsum(v[t] * v[t] for t in v))
    return v, v_norm
    
def cosine(v1, v1_norm, v2, v2_norm):
    terms = set(v1.keys()).intersection(set(v2.keys()))
    inner_prod = math.fsum(v1[t] * v2[t] for t in terms)
    norm = v1_norm * v2_norm
    return inner_prod / norm if norm > 0 else 0

def jaccard(first, second):
    if first is None or second is None or len(first) == 0 or len(second) == 0:
        return 0
    return len(set(first).intersection(second)) / float(len(set(first).union(second)))


@F.udf(returnType=ArrayType(FloatType()))
def cosine_sim_to_target(target, others):
    if target is None or len(target) == 0 or others is None or len(others) == 0:
        return []
    cosine_ary = []
    text1_tmp = [clean_text(t) for t in target if t is not None]
    text1_cent = tokenize_text(text1_tmp)
    v1, v1_norm = to_vec(text1_cent)
    for other in others:
        cos_sim = 0
        if other is not None and len(other) > 0:
            text2_tmp = [clean_text(t) for t in other if t is not None]
            text2_cent = tokenize_text(text2_tmp)
            v2, v2_norm = to_vec(text2_cent)
            cos_sim = cosine(v1, v1_norm, v2, v2_norm)
        cosine_ary.append(cos_sim)
    return cosine_ary

@F.udf(returnType=ArrayType(FloatType()))
def jaccard_sim_to_target(target, others):
    if target is None or len(target) == 0 or others is None or len(others) == 0:
        return []
    jaccard_ary = []
    tokens_set1 = [clean_text(t) for t in target if t is not None]
    tokens_set1 = tokenize_text(tokens_set1)
    for other in others:
        jaccard_sim = 0
        if other is not None and len(other) > 0:
            tokens_set2 = [clean_text(t) for t in other if t is not None]
            tokens_set2 = tokenize_text(tokens_set2)
            jaccard_sim = jaccard(tokens_set1, tokens_set2)
        jaccard_ary.append(jaccard_sim)
    return jaccard_ary
    
    
@F.udf(returnType=ArrayType(FloatType()))
def array_norm(x):
    if x is None or len(x) == 0:
        return None
    norm_x = x / np.linalg.norm(x)
    return norm_x.tolist()
    
@F.udf(returnType=FloatType())
def array_percentile(x, percent):
    if x is None or len(x) == 0:
        return None
    x = [v for v in x if v is not None]
    if len(x) > 0:
        return float(np.percentile(x,percent))
    return -1
    
@F.udf(returnType=FloatType())
def score_ratio(minval, maxval):
    if minval is None or maxval is None:
        return None
    return float(maxval/minval if minval > 0 else 0)
    
array_mean = F.udf(lambda x: None if x is None or len(x) == 0 else float(np.mean(x)), FloatType())
array_var = F.udf(lambda x: None if x is None or len(x) == 0 else float(np.var(x)), FloatType())

@F.udf(returnType=FloatType())
def poisson_likelihood(mean):
    if mean is None:
        return 0
    return mean * math.exp(-mean)

# Define the UDF for calculating mean of an array
array_mean = F.udf(lambda x: None if x is None or len(x) == 0 else float(np.mean(x)), FloatType()) 