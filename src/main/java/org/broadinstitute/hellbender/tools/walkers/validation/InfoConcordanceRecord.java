package org.broadinstitute.hellbender.tools.walkers.validation;

import htsjdk.variant.variantcontext.VariantContext;
import org.broadinstitute.hellbender.exceptions.UserException;
import org.broadinstitute.hellbender.utils.tsv.DataLine;
import org.broadinstitute.hellbender.utils.tsv.TableColumnCollection;
import org.broadinstitute.hellbender.utils.tsv.TableReader;
import org.broadinstitute.hellbender.utils.tsv.TableWriter;

import java.io.File;
import java.io.IOException;

public class InfoConcordanceRecord {
    private static final String VARIANT_TYPE_COLUMN_NAME = "type";
    private static final String EVAL_INFO_KEY = "eval_info_key";
    private static final String TRUE_INFO_KEY = "true_info_key";
    private static final String MEAN_DIFFERENCE = "mean_difference";
    private static final String STD_DIFFERENCE = "std_difference";
    private static final String[] INFO_CONCORDANCE_COLUMN_HEADER =
            {VARIANT_TYPE_COLUMN_NAME, EVAL_INFO_KEY, TRUE_INFO_KEY, MEAN_DIFFERENCE, STD_DIFFERENCE};

    final VariantContext.Type type;
    final String evalKey;
    final String trueKey;
    final double mean;
    final double std;

    public InfoConcordanceRecord(final VariantContext.Type type, final String evalKey, final String trueKey, final double mean, final double std) {
        this.type = type;
        this.evalKey = evalKey;
        this.trueKey = trueKey;
        this.mean = mean;
        this.std = std;
    }
    public VariantContext.Type getVariantType() { return type; }

    public double getMean() {
        return mean;
    }

    public double getStd() {
        return std;
    }

    public String getEvalKey() {
        return evalKey;
    }

    public String getTrueKey() {
        return trueKey;
    }

    public static class InfoConcordanceWriter extends TableWriter<InfoConcordanceRecord> {
        private InfoConcordanceWriter(final File output) throws IOException {
            super(output, new TableColumnCollection(INFO_CONCORDANCE_COLUMN_HEADER));
        }

        @Override
        protected void composeLine(final InfoConcordanceRecord record, final DataLine dataLine) {
            dataLine.set(VARIANT_TYPE_COLUMN_NAME, record.getVariantType().toString())
                    .set(EVAL_INFO_KEY, record.getEvalKey())
                    .set(TRUE_INFO_KEY, record.getTrueKey())
                    .set(MEAN_DIFFERENCE, record.getMean())
                    .set(STD_DIFFERENCE, record.getStd());
        }

    }

    public static InfoConcordanceWriter getWriter(final File outputTable){
        try {
            InfoConcordanceWriter writer = new InfoConcordanceWriter(outputTable);
            return writer;
        } catch (IOException e){
            throw new UserException(String.format("Encountered an IO exception while reading from %s.", outputTable), e);
        }
    }

    public static class InfoConcordanceReader extends TableReader<InfoConcordanceRecord> {
        public InfoConcordanceReader(final File summary) throws IOException {
            super(summary);
        }

        @Override
        protected InfoConcordanceRecord createRecord(final DataLine dataLine) {
            final VariantContext.Type type = VariantContext.Type.valueOf(dataLine.get(VARIANT_TYPE_COLUMN_NAME));
            final String evalKey = dataLine.get(EVAL_INFO_KEY);
            final String trueKey = dataLine.get(TRUE_INFO_KEY);
            final double mean = Double.parseDouble(dataLine.get(MEAN_DIFFERENCE));
            final double std = Double.parseDouble(dataLine.get(STD_DIFFERENCE));

            return new InfoConcordanceRecord(type, evalKey, trueKey, mean, std);
        }
    }
}