package org.broadinstitute.hellbender.tools.spark.transforms.markduplicates;

import com.google.api.client.util.Lists;
import com.google.common.collect.ImmutableList;
import htsjdk.samtools.*;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.broadinstitute.hellbender.engine.spark.SparkContextFactory;
import org.broadinstitute.hellbender.engine.spark.datasources.ReadsSparkSink;
import org.broadinstitute.hellbender.utils.read.*;
import org.broadinstitute.hellbender.utils.read.markduplicates.MarkDuplicatesScoringStrategy;
import org.broadinstitute.hellbender.utils.read.markduplicates.OpticalDuplicateFinder;
import org.broadinstitute.hellbender.utils.read.markduplicates.ReadsKey;
import org.broadinstitute.hellbender.GATKBaseTest;
import org.broadinstitute.hellbender.utils.test.SamAssertionUtils;
import org.testng.Assert;
import org.testng.annotations.Test;
import scala.Tuple2;
import scala.collection.Seq;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

public class MarkDuplicatesSparkUtilsUnitTest extends GATKBaseTest {
    @Test(groups = "spark")
    public void testSpanningIterator() {
        check(Collections.emptyIterator(), Collections.emptyList());
        check(ImmutableList.of(pair(1, "a")).iterator(),
                ImmutableList.of(pairIterable(1, "a")));
        check(ImmutableList.of(pair(1, "a"), pair(1, "b")).iterator(),
                ImmutableList.of(pairIterable(1, "a", "b")));
        check(ImmutableList.of(pair(1, "a"), pair(2, "b")).iterator(),
                ImmutableList.of(pairIterable(1, "a"), pairIterable(2, "b")));
        check(ImmutableList.of(pair(1, "a"), pair(1, "b"), pair(2, "c")).iterator(),
                ImmutableList.of(pairIterable(1, "a", "b"), pairIterable(2, "c")));
        check(ImmutableList.of(pair(1, "a"), pair(2, "b"), pair(2, "c")).iterator(),
                ImmutableList.of(pairIterable(1, "a"), pairIterable(2, "b", "c")));
        check(ImmutableList.of(pair(1, "a"), pair(2, "b"), pair(1, "c")).iterator(),
                ImmutableList.of(pairIterable(1, "a"), pairIterable(2, "b"), pairIterable(1, "c")));
    }


    private String getReadGroupId(final SAMFileHeader header, final int index) {
        return header.getReadGroups().get(index).getReadGroupId();
    }

    private static <K, V> void check(Iterator<Tuple2<K, V>> it, List<Tuple2<K, Iterable<V>>> expected) {
        Iterator<Tuple2<K, Iterable<V>>> spanning = MarkDuplicatesSparkUtils.spanningIterator(it);
        ArrayList<Tuple2<K, Iterable<V>>> actual = Lists.newArrayList(spanning);
        Assert.assertEquals(actual, expected);
    }

    private static <K, V> Tuple2<K, V> pair(K key, V value) {
        return new Tuple2<>(key, value);
    }

    private static Tuple2<Integer, Iterable<String>> pairIterable(int i, String... s) {
        return new Tuple2<>(i, ImmutableList.copyOf(s));
    }

    private static Tuple2<String, Iterable<GATKRead>> pairIterable(String key, GATKRead... reads) {
        return new Tuple2<>(key, ImmutableList.copyOf(reads));
    }

    @Test
    // Test that asserts the duplicate marking is sorting agnostic, specifically this is testing that when reads are scrambled across
    // partitions in the input that all reads in a group are getting properly duplicate marked together as they are for queryname sorted bams
    public void testSortOrderParitioningCorrectness() throws IOException {

        JavaSparkContext ctx = SparkContextFactory.getTestSparkContext();
        JavaRDD<GATKRead> unsortedReads = generateUnsortedReads(10000,3, ctx, 100, true);
        JavaRDD<GATKRead> pariedEndsQueryGrouped = generateUnsortedReads(10000,3, ctx,1, false);

        SAMFileHeader unsortedHeader = hg19Header.clone();
        unsortedHeader.setSortOrder(SAMFileHeader.SortOrder.unsorted);
        SAMFileHeader sortedHeader = hg19Header.clone();
        sortedHeader.setSortOrder(SAMFileHeader.SortOrder.queryname);

        // Using the header flagged as unsorted will result in the reads being sorted again
        JavaRDD<GATKRead> unsortedReadsMarked = MarkDuplicatesSpark.mark(unsortedReads,unsortedHeader, MarkDuplicatesScoringStrategy.SUM_OF_BASE_QUALITIES,new OpticalDuplicateFinder(),100,true);
        JavaRDD<GATKRead> sortedReadsMarked = MarkDuplicatesSpark.mark(pariedEndsQueryGrouped,sortedHeader, MarkDuplicatesScoringStrategy.SUM_OF_BASE_QUALITIES,new OpticalDuplicateFinder(),1,true);

        Iterator<GATKRead> sortedReadsFinal = sortedReadsMarked.sortBy(GATKRead::commonToString, false, 1).collect().iterator();
        Iterator<GATKRead> unsortedReadsFinal = unsortedReadsMarked.sortBy(GATKRead::commonToString, false, 1).collect().iterator();

        // Comparing the output reads to ensure they are all duplicate marked correctly
        while (sortedReadsFinal.hasNext()) {
            GATKRead read1 = sortedReadsFinal.next();
            GATKRead read2 = unsortedReadsFinal.next();
            Assert.assertEquals(read1.getName(), read2.getName());
            Assert.assertEquals(read1.isDuplicate(), read2.isDuplicate());
        }
    }

    private JavaRDD<GATKRead> generateUnsortedReads(int numReadGroups, int numDuplicatesPerGroup, JavaSparkContext ctx, int numPartitions, boolean coordinate) {
        int readNameCounter = 0;
        SAMRecordSetBuilder samRecordSetBuilder = new SAMRecordSetBuilder(true, SAMFileHeader.SortOrder.coordinate,
                true, SAMRecordSetBuilder.DEFAULT_CHROMOSOME_LENGTH, SAMRecordSetBuilder.DEFAULT_DUPLICATE_SCORING_STRATEGY);

        Random rand = new Random(10);
        for (int i = 0; i < numReadGroups; i++ ) {
            int start1 = rand.nextInt(SAMRecordSetBuilder.DEFAULT_CHROMOSOME_LENGTH);
            int start2 = rand.nextInt(SAMRecordSetBuilder.DEFAULT_CHROMOSOME_LENGTH);
            for (int j = 0; j < numDuplicatesPerGroup; j++) {
                samRecordSetBuilder.addPair("READ" + readNameCounter++, 0, start1, start2);
            }
        }
        final ReadCoordinateComparator coordinateComparitor = new ReadCoordinateComparator(hg19Header);
        List<SAMRecord> records = Lists.newArrayList(samRecordSetBuilder.getRecords());
        if (coordinate) {
            records.sort(new SAMRecordCoordinateComparator());
        } else {
            records.sort(new SAMRecordQueryNameComparator());
        }

        return ctx.parallelize(records, numPartitions).map(SAMRecordToGATKReadAdapter::new);
    }

}
