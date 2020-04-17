#!/usr/bin/env Rscript

library(gdsfmt)
library(SNPRelate)

args <- commandArgs(trailingOnly=TRUE)

if (length(args) < 4) {
  stop("usage: <bed_file> <bim_file> <fam_file> <out>", call.=FALSE)
}

snpgdsBED2GDS(bed.fn=args[1], bim.fn=args[2], fam.fn=args[3], out.gdsfn=args[4])
