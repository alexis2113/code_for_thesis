import numpy as np
import json
import os

# Here we define a few function to perform the caculation of
# energy distance
# A example implementing the methods suggest by Xiao .et.al in his article
#"Multivariate global sensitivity analysis for dynamic models based on energy distance"


# partitions of dataset based on parameters 
with open("../partitions_row_index.json","rb") as f:
        allglist:list=json.load(f)

def countrow(df):
        nrows=df.shape[0]
        return nrows


def totaldis(df):
        total=df.distance.sum()
        nrows=countrow(df)
        total=total/nrows**2
        return total

def find1(df,group):
        member=df.loc[df["rowi"].isin(group)] 
        return member


def find2(df,group):
        member=find1(df,group)
        member=member.loc[df["rowj"].isin(group)]
        return member

# distance within groups
def intra(df,group):
        member=find1(df,group)
        gz=len(member)
        if gz >2:
                B=member.distance.sum()
                B=B*2
                B=B/gz**2
                return B
        else:
                return 0


def summember(df,group):
        member=find2(df,group)
        groupz=len(member)
        if groupz > 2:
                A=member.distance.sum()
                A=A*2
                nrows=countrow(df)
                A=2*A/groupz*nrows
                return A
        else:
                return 0


def bygroup(df,group):
  A=summember(df,group)
  B=intra(df,group)
  C=totaldis(df)
  res=A-B-C
  return res



# setup batch for parallel process
def batch(df,seq):
         sub_results = []
         for x in seq:
                 sub_results.append(bygroup(df,x))
         return sub_results

