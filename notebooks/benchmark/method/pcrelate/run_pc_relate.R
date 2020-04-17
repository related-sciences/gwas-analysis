#!/usr/bin/env Rscript

library(gdsfmt)
library(SNPRelate)
library(GWASTools)
library(GENESIS)
library(tictoc)

args <- commandArgs(trailingOnly=TRUE)

if (length(args) < 1) {
  stop("usage: <gds_file>", call.=FALSE)
}

gds_filepath = args[1]

tic("KING kinship")
# we compute KING kinship estimator using R,
# we could also use king software itself
# TODO: check if we need to include any thresholds
genofile <- snpgdsOpen(gds_filepath)
king_mat <- snpgdsIBDKING(genofile, num.thread=8)
# Prepare for pcair structure
king_mat_2 = kingToMatrix(king_mat)
toc(log = TRUE)
snpgdsClose(genofile)

reader <- GdsGenotypeReader(gds_filepath, "snp,scan")
geno_data <- GenotypeData(reader)

tic("PC-AIR")
# TODO: we don't include: snp.include = pruned, unrel.set
pcair_result <- pcair(geno_data,
                      kinobj = king_mat_2,
                      divobj = king_mat_2)
toc(log = TRUE)
summary(pcair_result)

# plot top 2 PCs
plot(pcair_result)
# plot PCs 3 and 4
plot(pcair_result, vx = 3, vy = 4)

# TODO: no pruning
geno_data <- GenotypeBlockIterator(geno_data)
tic("PC-Relate")
pcrelate_result <- pcrelate(geno_data,
                            pcs = pcair_result$vectors[,1:2],
                            training.set = pcair_result$unrels)
toc(log = TRUE)

summary(pcrelate_result)
#plot(pcrelate_result$kinBtwn$k0, pcrelate_result$kinBtwn$kin, xlab="k0", ylab="kinship")
logs <- tic.log(format = TRUE)
lapply(logs, write, "times.txt", append=TRUE)
