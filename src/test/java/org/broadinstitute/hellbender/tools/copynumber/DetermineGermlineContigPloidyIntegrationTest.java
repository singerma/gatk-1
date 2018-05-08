package org.broadinstitute.hellbender.tools.copynumber;

import htsjdk.samtools.SAMSequenceDictionary;
import htsjdk.samtools.SAMSequenceRecord;
import org.apache.commons.math3.random.RandomDataGenerator;
import org.broadinstitute.hellbender.CommandLineProgramTest;
import org.broadinstitute.hellbender.cmdline.StandardArgumentDefinitions;
import org.broadinstitute.hellbender.exceptions.UserException;
import org.broadinstitute.hellbender.tools.copynumber.arguments.CopyNumberStandardArgument;
import org.broadinstitute.hellbender.tools.copynumber.formats.collections.SimpleCountCollection;
import org.broadinstitute.hellbender.tools.copynumber.formats.metadata.SimpleSampleLocatableMetadata;
import org.broadinstitute.hellbender.tools.copynumber.formats.records.SimpleCount;
import org.broadinstitute.hellbender.utils.SimpleInterval;
import org.broadinstitute.hellbender.utils.Utils;
import org.broadinstitute.hellbender.utils.param.ParamUtils;
import org.broadinstitute.hellbender.utils.test.ArgumentsBuilder;
import org.testng.annotations.Test;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Integration tests for {@link DetermineGermlineContigPloidy}.
 *
 * The test runs the CLI tool in Cohort and Case run-modes on a small simulated data.
 *
 */
public final class DetermineGermlineContigPloidyIntegrationTest extends CommandLineProgramTest {
    private static final String SIMULATED_DATA_DIR = toolsTestDir + "copynumber/gcnv-sim-data/";
    private static final File PLOIDY_STATE_PRIORS_FILE =
            new File(SIMULATED_DATA_DIR, "ploidy_state_priors.tsv");
    private static final List<File> COUNT_FILES = Arrays.asList(
//            new File("/home/slee/working/gatk/test_files/aneuploidy-samples/cases/10C110552.cram.counts.hdf5"),
//            new File("/home/slee/working/gatk/test_files/aneuploidy-samples/cases/10C112547.cram.counts.hdf5"),
//            new File("/home/slee/working/gatk/test_files/aneuploidy-samples/cases/11C119003.cram.counts.hdf5"),
//            new File("/home/slee/working/gatk/test_files/aneuploidy-samples/cases/11C120584.cram.counts.hdf5"),
//            new File("/home/slee/working/gatk/test_files/aneuploidy-samples/cases/11C122687.cram.counts.hdf5"),
//            new File("/home/slee/working/gatk/test_files/aneuploidy-samples/cases/8007540246.cram.counts.hdf5"),
//            new File("/home/slee/working/gatk/test_files/aneuploidy-samples/cases/8007540249.cram.counts.hdf5"),
//            new File("/home/slee/working/gatk/test_files/aneuploidy-samples/cases/8007540504.cram.counts.hdf5"),
//            new File("/home/slee/working/gatk/test_files/aneuploidy-samples/cases/8007540650.cram.counts.hdf5"),
//            new File("/home/slee/working/gatk/test_files/aneuploidy-samples/cases/8007540842.cram.counts.hdf5"),
//            new File("/home/slee/working/gatk/test_files/aneuploidy-samples/cases/8007543000.cram.counts.hdf5"),
//            new File("/home/slee/working/gatk/test_files/aneuploidy-samples/cases/A-WCAP-WC000799-BL-COL-50949BL1.cram.counts.hdf5"),
//            new File("/home/slee/working/gatk/test_files/aneuploidy-samples/cases/Ger_2392.cram.counts.hdf5"),
//            new File("/home/slee/working/gatk/test_files/aneuploidy-samples/cases/Ger_2791.cram.counts.hdf5"),
//            new File("/home/slee/working/gatk/test_files/aneuploidy-samples/cases/Ger_3154.cram.counts.hdf5"),
//            new File("/home/slee/working/gatk/test_files/aneuploidy-samples/cases/Ger_3277.cram.counts.hdf5"),
//            new File("/home/slee/working/gatk/test_files/aneuploidy-samples/cases/Ger_907.cram.counts.hdf5"),
//            new File("/home/slee/working/gatk/test_files/aneuploidy-samples/cases/MH0129914.cram.counts.hdf5"),
//            new File("/home/slee/working/gatk/test_files/aneuploidy-samples/cases/MH0136494.cram.counts.hdf5"),
//            new File("/home/slee/working/gatk/test_files/aneuploidy-samples/cases/MH0143011.cram.counts.hdf5"),
//            new File("/home/slee/working/gatk/test_files/aneuploidy-samples/cases/NWD146665.cram.counts.hdf5"),
//            new File("/home/slee/working/gatk/test_files/aneuploidy-samples/cases/NWD208455.cram.counts.hdf5"),
//            new File("/home/slee/working/gatk/test_files/aneuploidy-samples/cases/NWD394702.cram.counts.hdf5"),
//            new File("/home/slee/working/gatk/test_files/aneuploidy-samples/cases/NWD411318.cram.counts.hdf5"),
//            new File("/home/slee/working/gatk/test_files/aneuploidy-samples/cases/NWD490973.cram.counts.hdf5"),
//            new File("/home/slee/working/gatk/test_files/aneuploidy-samples/cases/NWD586820.cram.counts.hdf5"),
//            new File("/home/slee/working/gatk/test_files/aneuploidy-samples/cases/NWD744200.cram.counts.hdf5"),
//            new File("/home/slee/working/gatk/test_files/aneuploidy-samples/cases/NWD789325.cram.counts.hdf5"),
//            new File("/home/slee/working/gatk/test_files/aneuploidy-samples/cases/NWD847295.cram.counts.hdf5"),
//            new File("/home/slee/working/gatk/test_files/aneuploidy-samples/cases/NWD932860.cram.counts.hdf5"),
//            new File("/home/slee/working/gatk/test_files/aneuploidy-samples/cases/UK568-5.cram.counts.hdf5"),
//            new File("/home/slee/working/gatk/test_files/aneuploidy-samples/cases/UK75-2.cram.counts.hdf5"),
//            new File("/home/slee/working/gatk/test_files/aneuploidy-samples/cases/V11538.cram.counts.hdf5"),
//            new File("/home/slee/working/gatk/test_files/aneuploidy-samples/cases/V4-2446.cram.counts.hdf5"),
//            new File("/home/slee/working/gatk/test_files/aneuploidy-samples/cases/V4-2807.cram.counts.hdf5"),
//            new File("/home/slee/working/gatk/test_files/aneuploidy-samples/cases/V4-3438.cram.counts.hdf5"),
//            new File("/home/slee/working/gatk/test_files/aneuploidy-samples/cases/V4-3597.cram.counts.hdf5"),
//            new File("/home/slee/working/gatk/test_files/aneuploidy-samples/cases/VIR_1025.cram.counts.hdf5"),
//            new File("/home/slee/working/gatk/test_files/aneuploidy-samples/cases/VIR_1861.cram.counts.hdf5"),
//            new File("/home/slee/working/gatk/test_files/aneuploidy-samples/cases/VIR_2657.cram.counts.hdf5"),
            new File("/home/slee/working/gatk/test_files/aneuploidy-samples/panel/8007540135.cram.counts.hdf5")//,
//            new File("/home/slee/working/gatk/test_files/aneuploidy-samples/panel/8007540159.cram.counts.hdf5"),
//            new File("/home/slee/working/gatk/test_files/aneuploidy-samples/panel/8007540160.cram.counts.hdf5"),
//            new File("/home/slee/working/gatk/test_files/aneuploidy-samples/panel/8007540172.cram.counts.hdf5"),
//            new File("/home/slee/working/gatk/test_files/aneuploidy-samples/panel/8007540173.cram.counts.hdf5"),
//            new File("/home/slee/working/gatk/test_files/aneuploidy-samples/panel/8007540177.cram.counts.hdf5"),
//            new File("/home/slee/working/gatk/test_files/aneuploidy-samples/panel/8007540185.cram.counts.hdf5"),
//            new File("/home/slee/working/gatk/test_files/aneuploidy-samples/panel/8007540189.cram.counts.hdf5"),
//            new File("/home/slee/working/gatk/test_files/aneuploidy-samples/panel/8007540194.cram.counts.hdf5"),
//            new File("/home/slee/working/gatk/test_files/aneuploidy-samples/panel/8007540206.cram.counts.hdf5"),
//            new File("/home/slee/working/gatk/test_files/aneuploidy-samples/panel/8007540235.cram.counts.hdf5"),
//            new File("/home/slee/working/gatk/test_files/aneuploidy-samples/panel/8007540251.cram.counts.hdf5"),
//            new File("/home/slee/working/gatk/test_files/aneuploidy-samples/panel/8007540264.cram.counts.hdf5"),
//            new File("/home/slee/working/gatk/test_files/aneuploidy-samples/panel/8007540273.cram.counts.hdf5"),
//            new File("/home/slee/working/gatk/test_files/aneuploidy-samples/panel/8007540275.cram.counts.hdf5"),
//            new File("/home/slee/working/gatk/test_files/aneuploidy-samples/panel/8007540285.cram.counts.hdf5"),
//            new File("/home/slee/working/gatk/test_files/aneuploidy-samples/panel/8007540288.cram.counts.hdf5"),
//            new File("/home/slee/working/gatk/test_files/aneuploidy-samples/panel/8007540306.cram.counts.hdf5"),
//            new File("/home/slee/working/gatk/test_files/aneuploidy-samples/panel/8007540325.cram.counts.hdf5"),
//            new File("/home/slee/working/gatk/test_files/aneuploidy-samples/panel/8007540328.cram.counts.hdf5")
    );
    private static final File OUTPUT_DIR = createTempDir("test-ploidy");

    private static final class PloidyProfile {
        private final LinkedHashMap<String, Integer> contigToPloidyMap;

        private PloidyProfile(final List<String> contigs,
                              final List<Integer> ploidies) {
            Utils.validateArg(contigs.size() == ploidies.size(),
                    "Number of contigs and number of ploidies must be equal.");
            Utils.validateArg(contigs.stream().noneMatch(String::isEmpty),
                    "Contig names cannot be empty.");
            Utils.validateArg(ploidies.stream().allMatch(p -> p >= 0),
                    "Ploidies must be non-negative.");
            contigToPloidyMap = new LinkedHashMap<>(contigs.size());
            IntStream.range(0, contigs.size())
                    .forEach(i -> contigToPloidyMap.put(contigs.get(i), ploidies.get(i)));
        }

        private List<String> getContigs() {
            return new ArrayList<>(contigToPloidyMap.keySet());
        }

        private int getPloidy(final String contig) {
            return contigToPloidyMap.get(contig);
        }
    }

    private static final class SimulatedData {
        private static final int RANDOM_SEED = 1;
        private static final double MAPPING_ERROR = 0.01;

        private final List<PloidyProfile> ploidyProfiles;
        private final List<SimpleCountCollection> countCollections;

        private SimulatedData(final List<PloidyProfile> ploidyProfiles,
                              final double averageDepth,
                              final int numIntervalsPerContig) {
            this.ploidyProfiles = Utils.nonEmpty(ploidyProfiles);
            ParamUtils.isPositive(averageDepth, "Average depth must be positive.");
            ParamUtils.isPositive(numIntervalsPerContig, "Number of intervals per contig must be positive.");
            Utils.validateArg(ploidyProfiles.stream().map(PloidyProfile::getContigs).distinct().count() == 1,
                    "Ploidy profiles must all have same contigs.");

            final SAMSequenceDictionary sequenceDictionary = new SAMSequenceDictionary(
                    ploidyProfiles.get(0).getContigs().stream()
                            .map(c -> new SAMSequenceRecord(c, numIntervalsPerContig + 1))
                            .collect(Collectors.toList()));

            final RandomDataGenerator rng = new RandomDataGenerator();
            rng.reSeed(RANDOM_SEED);
            countCollections = IntStream.range(0, ploidyProfiles.size()).boxed()
                    .map(i -> generateCounts(
                            new SimpleSampleLocatableMetadata(String.format("sample_%d", i), sequenceDictionary),
                            ploidyProfiles.get(i), averageDepth, numIntervalsPerContig, rng))
                    .collect(Collectors.toList());
        }

        private List<File> writeCountFiles() {
            final List<File> countFiles = new ArrayList<>(countCollections.size());
            countCollections.forEach(c -> {
                final File outputFile = createTempFile(c.getMetadata().getSampleName(), ".tsv");
                c.write(outputFile);
                countFiles.add(outputFile);
            });
            return countFiles;
        }

        private static SimpleCountCollection generateCounts(final SimpleSampleLocatableMetadata metadata,
                                                            final PloidyProfile ploidyProfile,
                                                            final double averageDepth,
                                                            final int numIntervalsPerContig,
                                                            final RandomDataGenerator rng) {
            final List<SimpleCount> counts = ploidyProfile.getContigs().stream()
                    .map(c -> IntStream.range(1, numIntervalsPerContig + 1).boxed()
                            .map(i -> new SimpleCount(
                                    new SimpleInterval(c, i, i),
                                    (int) rng.nextPoisson(Math.max(ploidyProfile.getPloidy(c), MAPPING_ERROR) * averageDepth)))
                            .collect(Collectors.toList()))
                    .flatMap(List::stream)
                    .collect(Collectors.toList());
            return new SimpleCountCollection(metadata, counts);
        }
    }

    @Test(groups = {"python"})
    public void testCohort() {
        final List<String> contigs = Arrays.asList("1", "2", "3", "4", "5", "X", "Y");
        final SimulatedData simulatedData = new SimulatedData(
                Arrays.asList(
                        new PloidyProfile(contigs, Arrays.asList(2, 2, 2, 3, 2, 1, 1)),
                        new PloidyProfile(contigs, Arrays.asList(2, 2, 2, 2, 2, 2, 0))//,
//                        new PloidyProfile(contigs, Arrays.asList(2, 2, 2, 2, 2, 1, 1)),
//                        new PloidyProfile(contigs, Arrays.asList(2, 2, 2, 2, 2, 2, 0)),
//                        new PloidyProfile(contigs, Arrays.asList(2, 2, 2, 2, 2, 1, 1)),
//                        new PloidyProfile(contigs, Arrays.asList(2, 2, 2, 2, 2, 2, 0)),
//                        new PloidyProfile(contigs, Arrays.asList(2, 2, 2, 2, 2, 1, 1)),
//                        new PloidyProfile(contigs, Arrays.asList(2, 2, 2, 2, 2, 2, 0)),
//                        new PloidyProfile(contigs, Arrays.asList(2, 2, 2, 2, 2, 1, 1)),
//                        new PloidyProfile(contigs, Arrays.asList(2, 3, 2, 2, 2, 2, 0)),
//                        new PloidyProfile(contigs, Arrays.asList(2, 2, 2, 2, 2, 2, 0)),
//                        new PloidyProfile(contigs, Arrays.asList(2, 2, 2, 2, 2, 2, 0)),
//                        new PloidyProfile(contigs, Arrays.asList(2, 2, 2, 2, 2, 2, 0)),
//                        new PloidyProfile(contigs, Arrays.asList(2, 2, 2, 2, 2, 2, 0)),
//                        new PloidyProfile(contigs, Arrays.asList(2, 2, 2, 2, 2, 2, 0)),
//                        new PloidyProfile(contigs, Arrays.asList(2, 2, 2, 2, 2, 2, 0)),
//                        new PloidyProfile(contigs, Arrays.asList(2, 2, 2, 2, 2, 1, 2)),
//                        new PloidyProfile(contigs, Arrays.asList(2, 2, 2, 2, 2, 1, 0)),
//                        new PloidyProfile(contigs, Arrays.asList(2, 2, 2, 2, 2, 2, 1)),
//                        new PloidyProfile(contigs, Arrays.asList(2, 2, 2, 2, 2, 3, 0))
                ),
                100.,
                10000);
        final List<File> countFiles = simulatedData.writeCountFiles();

        final ArgumentsBuilder argsBuilder = new ArgumentsBuilder();
        countFiles.forEach(argsBuilder::addInput);
        argsBuilder.addFileArgument(DetermineGermlineContigPloidy.PLOIDY_STATE_PRIORS_FILE_LONG_NAME, PLOIDY_STATE_PRIORS_FILE)
//                .addArgument(StandardArgumentDefinitions.OUTPUT_LONG_NAME, OUTPUT_DIR.getAbsolutePath())
                .addArgument(StandardArgumentDefinitions.OUTPUT_LONG_NAME, "/home/slee/working/gatk/test_files")
                .addArgument(CopyNumberStandardArgument.OUTPUT_PREFIX_LONG_NAME, "test-ploidy-cohort")
                .addArgument(DetermineGermlineContigPloidy.MAXIMUM_COUNT_LONG_NAME, "1000")
                .addArgument(StandardArgumentDefinitions.VERBOSITY_NAME, "DEBUG");
        runCommandLine(argsBuilder);
    }

    @Test(groups = {"python"})
    public void testAneuploidyCohort() {
        final ArgumentsBuilder argsBuilder = new ArgumentsBuilder();
        COUNT_FILES.forEach(argsBuilder::addInput);
        argsBuilder.addFileArgument(DetermineGermlineContigPloidy.PLOIDY_STATE_PRIORS_FILE_LONG_NAME, new File("/home/slee/working/gatk/test_files/ploidy_state_priors.tsv"))
                .addArgument(StandardArgumentDefinitions.OUTPUT_LONG_NAME, "/home/slee/working/gatk/test_files")
                .addArgument(CopyNumberStandardArgument.OUTPUT_PREFIX_LONG_NAME, "test-aneuploidy-cohort")
                .addArgument(DetermineGermlineContigPloidy.MAXIMUM_COUNT_LONG_NAME, "1000")
                .addArgument(StandardArgumentDefinitions.VERBOSITY_NAME, "DEBUG");
        runCommandLine(argsBuilder);
    }

    @Test(groups = {"python"}, expectedExceptions = UserException.BadInput.class)
    public void testCohortWithoutContigPloidyPriors() {
        final ArgumentsBuilder argsBuilder = new ArgumentsBuilder();
        COUNT_FILES.forEach(argsBuilder::addInput);
        argsBuilder.addArgument(StandardArgumentDefinitions.OUTPUT_LONG_NAME, OUTPUT_DIR.getAbsolutePath())
                .addArgument(CopyNumberStandardArgument.OUTPUT_PREFIX_LONG_NAME, "test-ploidy-cohort")
                .addArgument(StandardArgumentDefinitions.VERBOSITY_NAME, "DEBUG");
        runCommandLine(argsBuilder);
    }

    @Test(groups = {"python"}, expectedExceptions = UserException.BadInput.class)
    public void testCohortWithSingleSample() {
        final ArgumentsBuilder argsBuilder = new ArgumentsBuilder();
        argsBuilder.addInput(COUNT_FILES.get(0));
        argsBuilder.addArgument(StandardArgumentDefinitions.OUTPUT_LONG_NAME, OUTPUT_DIR.getAbsolutePath())
                .addArgument(CopyNumberStandardArgument.OUTPUT_PREFIX_LONG_NAME, "test-ploidy-cohort")
                .addArgument(StandardArgumentDefinitions.VERBOSITY_NAME, "DEBUG");
        runCommandLine(argsBuilder);
    }

    @Test(groups = {"python"}, expectedExceptions = IllegalArgumentException.class)
    public void testCohortDuplicateFiles() {
        final ArgumentsBuilder argsBuilder = new ArgumentsBuilder();
        COUNT_FILES.forEach(argsBuilder::addInput);
        argsBuilder.addInput(COUNT_FILES.get(0));  //duplicate
        argsBuilder.addFileArgument(DetermineGermlineContigPloidy.PLOIDY_STATE_PRIORS_FILE_LONG_NAME, PLOIDY_STATE_PRIORS_FILE)
                .addArgument(StandardArgumentDefinitions.OUTPUT_LONG_NAME, OUTPUT_DIR.getAbsolutePath())
                .addArgument(CopyNumberStandardArgument.OUTPUT_PREFIX_LONG_NAME, "test-ploidy-cohort")
                .addArgument(StandardArgumentDefinitions.VERBOSITY_NAME, "DEBUG");
        runCommandLine(argsBuilder);
    }

    /**
     * Use the first 5 samples as case and use the contig-ploidy model generated by {@link #testCohort()}
     */
    @Test(groups = {"python"}, dependsOnMethods = "testCohort")
    public void testCase() {
        final ArgumentsBuilder argsBuilder = new ArgumentsBuilder();
        COUNT_FILES.subList(0, 5).forEach(argsBuilder::addInput);
        argsBuilder.addArgument(StandardArgumentDefinitions.OUTPUT_LONG_NAME, OUTPUT_DIR.getAbsolutePath())
                .addArgument(CopyNumberStandardArgument.OUTPUT_PREFIX_LONG_NAME, "test-ploidy-case")
                .addArgument(CopyNumberStandardArgument.MODEL_LONG_NAME,
                        new File(OUTPUT_DIR, "test-ploidy-cohort-model").getAbsolutePath())
                .addArgument(StandardArgumentDefinitions.VERBOSITY_NAME, "DEBUG");
        runCommandLine(argsBuilder);
    }

    @Test(groups = {"python"}, dependsOnMethods = "testCohort", expectedExceptions = UserException.BadInput.class)
    public void testCaseWithContigPloidyPrior() {
        final ArgumentsBuilder argsBuilder = new ArgumentsBuilder();
        COUNT_FILES.subList(0, 5).forEach(argsBuilder::addInput);
        argsBuilder.addArgument(StandardArgumentDefinitions.OUTPUT_LONG_NAME, OUTPUT_DIR.getAbsolutePath())
                .addFileArgument(DetermineGermlineContigPloidy.PLOIDY_STATE_PRIORS_FILE_LONG_NAME, PLOIDY_STATE_PRIORS_FILE)
                .addArgument(CopyNumberStandardArgument.OUTPUT_PREFIX_LONG_NAME, "test-ploidy-case")
                .addArgument(CopyNumberStandardArgument.MODEL_LONG_NAME,
                        new File(OUTPUT_DIR, "test-ploidy-cohort-model").getAbsolutePath())
                .addArgument(StandardArgumentDefinitions.VERBOSITY_NAME, "DEBUG");
        runCommandLine(argsBuilder);
    }
}