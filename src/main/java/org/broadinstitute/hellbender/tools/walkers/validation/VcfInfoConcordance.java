package org.broadinstitute.hellbender.tools.walkers.validation;

import htsjdk.variant.variantcontext.VariantContext;
import htsjdk.variant.variantcontext.VariantContextBuilder;
import htsjdk.variant.variantcontext.writer.VariantContextWriter;
import htsjdk.variant.vcf.*;
import org.apache.commons.collections4.Predicate;
import org.apache.commons.lang.mutable.MutableLong;
import org.broadinstitute.barclay.argparser.Argument;
import org.broadinstitute.barclay.argparser.BetaFeature;
import org.broadinstitute.barclay.argparser.CommandLineProgramProperties;
import org.broadinstitute.barclay.help.DocumentedFeature;
import org.broadinstitute.hellbender.engine.AbstractConcordanceWalker;
import org.broadinstitute.hellbender.engine.ReadsContext;
import org.broadinstitute.hellbender.engine.ReferenceContext;
import org.broadinstitute.hellbender.exceptions.UserException;
import org.broadinstitute.hellbender.tools.walkers.variantutils.VariantsToTable;
import org.broadinstitute.hellbender.utils.Utils;
import org.broadinstitute.hellbender.utils.tsv.DataLine;
import org.broadinstitute.hellbender.utils.tsv.TableColumnCollection;
import org.broadinstitute.hellbender.utils.tsv.TableWriter;
import org.broadinstitute.hellbender.utils.variant.GATKVCFConstants;
import picard.cmdline.programgroups.VariantEvaluationProgramGroup;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Evaluate site-level concordance of an input VCF against a truth VCF.
 *
 * <p>This tool evaluates two variant callsets against each other and produces a six-column summary metrics table. The summary:</p>
 *
 * <ul>
 *     <li>stratifies SNP and INDEL calls,</li>
 *     <li>tallies true-positive, false-positive and false-negative calls,</li>
 *     <li>and calculates sensitivity and precision.</li>
 * </ul>
 *
 * <p>The tool assumes all records in the --truth VCF are passing truth variants. For the -eval VCF, the tool uses only unfiltered passing calls.</p>
 *
 * <p>Optionally, the tool can be set to produce VCFs of the following variant records, annotated with each variant's concordance status:</p>
 * <ul>
 *     <li>True positives and false negatives (i.e. all variants in the truth VCF): useful for calculating sensitivity</li>
 *     <li>True positives and false positives (i.e. all variants in the eval VCF): useful for obtaining a training data
 *     set for machine learning classifiers of artifacts</li>
 * </ul>
 *
 * <p>These output VCFs can be passed to {@link VariantsToTable} to produce a TSV file for statistical analysis in R
 * or Python.</p>
 *
 * <h3>Usage example</h3>
 *
 * <pre>
 * gatk VcfInfoConcordance \
 *   -R reference.fa \
 *   -eval eval.vcf \
 *   --truth truth.vcf \
 *   --summary summary.tsv
 * </pre>
 *
 */

@CommandLineProgramProperties(
        summary = VcfInfoConcordance.USAGE_SUMMARY,
        oneLineSummary = VcfInfoConcordance.USAGE_ONE_LINE_SUMMARY,
        programGroup = VariantEvaluationProgramGroup.class
)
@DocumentedFeature
@BetaFeature
public class VcfInfoConcordance extends AbstractConcordanceWalker {

    static final String USAGE_ONE_LINE_SUMMARY = "Evaluate concordance of info fields in an input VCF against a validated truth VCF";
    static final String USAGE_SUMMARY = "This tool evaluates info fields from an input VCF against a VCF that has been validated" +
            " and is considered to represent ground truth.\n";

    public static final String SUMMARY_LONG_NAME = "summary";
    public static final String SUMMARY_SHORT_NAME = "S";

    @Argument(doc = "A table of summary statistics (true positives, sensitivity, etc.)",
            fullName = SUMMARY_LONG_NAME,
            shortName = SUMMARY_SHORT_NAME)
    protected File summary;

    @Argument(fullName = "eval-info-key", shortName = "eval-info-key", doc = "Info key from eval vcf", optional = true)
    protected String evalInfoKey = GATKVCFConstants.CNN_2D_KEY;

    @Argument(fullName = "truth-info-key", shortName = "truth-info-key", doc = "Info key from truth vcf", optional = true)
    protected String truthInfoKey = GATKVCFConstants.CNN_2D_KEY;

    @Argument(fullName = "epsilon", shortName = "epsilon", doc = "Difference tolerance", optional = true)
    protected double epsilon = 0.1;

    // we count true positives, false positives, false negatives for snps and indels
    private final EnumMap<ConcordanceState, MutableLong> snpCounts = new EnumMap<>(ConcordanceState.class);
    private final EnumMap<ConcordanceState, MutableLong> indelCounts = new EnumMap<>(ConcordanceState.class);

    private double snpSumDelta = 0;
    private double snpSumDeltaSquared = 0;
    private double indelSumDelta = 0;
    private double indelSumDeltaSquared = 0;

    @Override
    public void onTraversalStart() {
        for (final ConcordanceState state : ConcordanceState.values()) {
            snpCounts.put(state, new MutableLong(0));
            indelCounts.put(state, new MutableLong(0));
        }
    }

    @Override
    protected void apply(final TruthVersusEval truthVersusEval, final ReadsContext readsContext, final ReferenceContext refContext) {
        final ConcordanceState concordanceState = truthVersusEval.getConcordance();
        if (truthVersusEval.getTruthIfPresentElseEval().isSNP()) {
            snpCounts.get(concordanceState).increment();
        } else {
            indelCounts.get(concordanceState).increment();
        }

        switch (concordanceState) {
            case TRUE_POSITIVE:
                infoDifference(truthVersusEval.getEval(), truthVersusEval.getTruth());
                break;

            case FALSE_POSITIVE:
            case FALSE_NEGATIVE:
            case FILTERED_TRUE_NEGATIVE:
            case FILTERED_FALSE_NEGATIVE:
                break;

            default:
                throw new IllegalStateException("Unexpected ConcordanceState: " + concordanceState.toString());
        }

    }

    private void infoDifference(VariantContext eval, VariantContext truth){
        double evalVal = Double.valueOf((String)eval.getAttribute(evalInfoKey));
        double truthVal = Double.valueOf((String) truth.getAttribute(truthInfoKey));
        double delta = evalVal-truthVal;
        double deltaSquared = delta*delta;

        if (eval.isSNP()){
            snpSumDelta += Math.sqrt(deltaSquared);
            snpSumDeltaSquared += deltaSquared;

        } else if (eval.isIndel()) {
            indelSumDelta += Math.sqrt(deltaSquared);
            indelSumDeltaSquared += deltaSquared;
        }

        if (delta > epsilon){
            logger.warn(String.format("Difference (%f) greater than epsilon (%f) at %s:%d %s:",
                    delta, epsilon, eval.getContig(), eval.getStart(), eval.getAlleles().toString()));
            logger.warn(String.format("\t\tTruth info: "+ truth.getAttributes().toString()));
            logger.warn(String.format("\t\t Eval info: "+ eval.getAttributes().toString()));

        }

    }

    @Override
    public Object onTraversalSuccess() {
        double snpN =  snpCounts.get(ConcordanceState.TRUE_POSITIVE).doubleValue();
        double snpMean = snpSumDelta / snpN;
        double snpVariance = (snpSumDeltaSquared - (snpSumDelta * snpSumDelta) / snpN) / snpN;
        double snpStd = Math.sqrt(snpVariance);

        double indelN =  indelCounts.get(ConcordanceState.TRUE_POSITIVE).doubleValue();
        double indelMean = indelSumDelta / indelN;
        double indelVariance = (indelSumDeltaSquared - (indelSumDelta * indelSumDelta) / indelN) / indelN;
        double indelStd = Math.sqrt(indelVariance);

        logger.info(String.format("SNP average delta %f and standard deviation: %f", snpMean, snpStd));
        logger.info(String.format("INDEL average delta %f and standard deviation: %f", indelMean, indelStd));

        try (InfoConcordanceRecord.InfoConcordanceWriter concordanceWriter = InfoConcordanceRecord.getWriter(summary)){
            concordanceWriter.writeRecord(new InfoConcordanceRecord(VariantContext.Type.SNP, evalInfoKey, truthInfoKey, snpMean, snpStd));
            concordanceWriter.writeRecord(new InfoConcordanceRecord(VariantContext.Type.INDEL, evalInfoKey, truthInfoKey, indelMean, indelStd));
        } catch (IOException e){
            throw new UserException("Encountered an IO exception writing the concordance summary table", e);
        }

        return "SUCCESS";
    }

    @Override
    protected boolean areVariantsAtSameLocusConcordant(final VariantContext truth, final VariantContext eval) {
        final boolean sameRefAllele = truth.getReference().equals(eval.getReference());
        // we assume that the truth has a single alternate allele
        final boolean containsAltAllele = eval.getAlternateAlleles().contains(truth.getAlternateAllele(0));

        return sameRefAllele && containsAltAllele;
    }

    @Override
    protected Predicate<VariantContext> makeTruthVariantFilter() {
        return vc -> !vc.isFiltered() && ! vc.isSymbolicOrSV();
    }




}
