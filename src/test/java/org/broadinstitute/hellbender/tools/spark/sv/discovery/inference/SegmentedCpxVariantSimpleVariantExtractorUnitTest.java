package org.broadinstitute.hellbender.tools.spark.sv.discovery.inference;

import org.broadinstitute.hellbender.GATKBaseTest;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

import java.util.ArrayList;
import java.util.List;

public class SegmentedCpxVariantSimpleVariantExtractorUnitTest extends GATKBaseTest {

    @DataProvider(name = "forTestPairIterationWayOfReInterpretation")
    private Object[][] forTestPairIterationWayOfReInterpretation() {
        final List<Object[]> data = new ArrayList<>(20);

        return data.toArray(new Object[data.size()][]);
    }
    @Test(groups = "sv", dataProvider = "forTestPairIterationWayOfReInterpretation")
    public void testPairIterationWayOfReInterpretation() {

    }

    @DataProvider(name = "forTestIsConsistentWithCPX")
    private Object[][] forTestIsConsistentWithCPX() {
        final List<Object[]> data = new ArrayList<>(20);

        return data.toArray(new Object[data.size()][]);
    }
    @Test(groups = "sv", dataProvider = "forTestIsConsistentWithCPX")
    public void testIsConsistentWithCPX() {

    }

    @DataProvider(name = "forTestInversionConsistencyCheck")
    private Object[][] forTestInversionConsistencyCheck() {
        final List<Object[]> data = new ArrayList<>(20);

        return data.toArray(new Object[data.size()][]);
    }
    @Test(groups = "sv", dataProvider = "forTestInversionConsistencyCheck")
    public void testInversionConsistencyCheck() {

    }

    @DataProvider(name = "forTestDeletionConsistencyCheck")
    private Object[][] forTestDeletionConsistencyCheck() {
        final List<Object[]> data = new ArrayList<>(20);

        return data.toArray(new Object[data.size()][]);
    }
    @Test(groups = "sv", dataProvider = "forTestDeletionConsistencyCheck")
    public void testDeletionConsistencyCheck() {

    }

    @DataProvider(name = "forTestRemoveDuplicates")
    private Object[][] forTestRemoveDuplicates() {
        final List<Object[]> data = new ArrayList<>(20);

        return data.toArray(new Object[data.size()][]);
    }
    @Test(groups = "sv", dataProvider = "forTestRemoveDuplicates")
    public void testRemoveDuplicates() {

    }

    @DataProvider(name = "forTestMergeSameVariants")
    private Object[][] forTestMergeSameVariants() {
        final List<Object[]> data = new ArrayList<>(20);

        return data.toArray(new Object[data.size()][]);
    }
    @Test(groups = "sv", dataProvider = "forTestMergeSameVariants")
    public void testMergeSameVariants() {

    }

    //==================================================================================================================

    @DataProvider(name = "forTestGetInsFromOneEnd")
    private Object[][] forTestGetInsFromOneEnd() {
        final List<Object[]> data = new ArrayList<>(20);

        return data.toArray(new Object[data.size()][]);
    }
    @Test(groups = "sv", dataProvider = "forTestGetInsFromOneEnd")
    public void testGetInsFromOneEnd() {

    }

    @DataProvider(name = "forTestGetInsLen")
    private Object[][] forTestGetInsLen() {
        final List<Object[]> data = new ArrayList<>(20);

        return data.toArray(new Object[data.size()][]);
    }
    @Test(groups = "sv")
    public void testGetInsLen() {

    }

    @DataProvider(name = "forTestZeroAndOneSegmentCpxVariantExtractor")
    private Object[][] forTestZeroAndOneSegmentCpxVariantExtractor() {
        final List<Object[]> data = new ArrayList<>(20);

        return data.toArray(new Object[data.size()][]);
    }
    @Test(groups = "sv", dataProvider = "forTestZeroAndOneSegmentCpxVariantExtractor")
    public void testZeroAndOneSegmentCpxVariantExtractor() {

    }

    @DataProvider(name = "forTestMultiSegmentsCpxVariantExtractor")
    private Object[][] forTestMultiSegmentsCpxVariantExtractor() {
        final List<Object[]> data = new ArrayList<>(20);

        return data.toArray(new Object[data.size()][]);
    }
    @Test(groups = "sv", dataProvider = "forTestMultiSegmentsCpxVariantExtractor")
    public void testMultiSegmentsCpxVariantExtractor() {

    }
}
