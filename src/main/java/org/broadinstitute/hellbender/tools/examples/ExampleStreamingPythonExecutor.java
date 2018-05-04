package org.broadinstitute.hellbender.tools.examples;

import org.broadinstitute.barclay.argparser.Argument;
import org.broadinstitute.barclay.argparser.CommandLineProgramProperties;
import org.broadinstitute.hellbender.cmdline.StandardArgumentDefinitions;
import org.broadinstitute.hellbender.cmdline.programgroups.ExampleProgramGroup;
import org.broadinstitute.hellbender.engine.FeatureContext;
import org.broadinstitute.hellbender.engine.ReadWalker;
import org.broadinstitute.hellbender.engine.ReferenceContext;
import org.broadinstitute.hellbender.utils.python.StreamingPythonScriptExecutor;
import org.broadinstitute.hellbender.utils.read.GATKRead;
import org.broadinstitute.hellbender.utils.runtime.AsynchronousStreamWriter;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Example ReadWalker program that uses a Python streaming executor to stream summary data from a BAM
 * input file to a Python process through an asynchronous stream writer. Reads data is accumulated in
 * a List until a batch size threshold is reached, at which point the batch is handed off to the
 * asynchronous stream writer, which writes the batch to the FIFO stream on a background thread. The
 * Python process in turn just writes the data to an output file.
 *
 * <ol>
 * <li>Creates a StreamingPythonExecutor.</li>
 * <li>Creates an AsynchronousWriterService to allow writing to the stream in batches on a background thread</li>
 * <li>Writes a string of attributes for each read to the List until the batchSize threshold is reached.</li>
 * <li>Uses Python to read each attribute line from the FIFO, and write it to the output file.</li>
 * </ol>
 *
 * See https://github.com/broadinstitute/gatk/wiki/Writing-GATK-Tools-that-use-Python for more information
 * on using Python with GATK.
 */
@CommandLineProgramProperties(
        summary = "Example/toy program that uses a Python script.",
        oneLineSummary = "Example/toy program that uses a Python script.",
        programGroup = ExampleProgramGroup.class,
        omitFromCommandLine = true
)
public class ExampleStreamingPythonExecutor extends ReadWalker {
    private final static String NL = System.lineSeparator();

    @Argument(fullName = StandardArgumentDefinitions.OUTPUT_LONG_NAME,
            shortName = StandardArgumentDefinitions.OUTPUT_SHORT_NAME,
            doc = "Output file")
    private File outputFile; // output file produced by Python code

    @Argument(fullName = "batchSize",
            doc = "Size of a batch for writing")
    private int batchSize = 1000;

    // Create the Python executor. This doesn't actually start the Python process, but verifies that
    // the requested Python executable exists and can be located.
    final StreamingPythonScriptExecutor pythonExecutor = new StreamingPythonScriptExecutor(true);

    private AsynchronousStreamWriter<String> asyncStreamWriter = null;
    private List<String> batchList = new ArrayList<>(batchSize);
    private int batchCount = 0;

    @Override
    public void onTraversalStart() {

        // Start the Python process, and get a strream writer from the executor to use to send data to Python.
        pythonExecutor.start(Collections.emptyList());
        asyncStreamWriter = pythonExecutor.getStreamWriter(AsynchronousStreamWriter.stringSerializer);

        // Also, ask Python to open the output file, where it will write the contents of everything it reads
        // from the executor stream. <code sendSynchronousCommand/>
        pythonExecutor.sendSynchronousCommand(String.format("tempFile = open('%s', 'w')" + NL, outputFile.getAbsolutePath()));
    }

    @Override
    public void apply(GATKRead read, ReferenceContext referenceContext, FeatureContext featureContext ) {
        // Extract data from the read and accumulate, unless we've reached a batch size, in which case we
        // kick off an asynchronous batch write.
        if (batchCount == batchSize) {
            waitForPreviousBatchCompletion();   // wait for the last batch to complete, if there is one
            startAsynchronousBatchWrite();      // start a new batch
        }
        batchList.add(String.format(
                "Read at %s:%d-%d:\n%s\n",
                read.getContig(), read.getStart(), read.getEnd(), read.getBasesString()));
        batchCount++;
    }

    /**
     * On traversal success, write the remaining batch. Post traversal work would be done here.
     * @return Success indicator.
     */
    public Object onTraversalSuccess() {
        waitForPreviousBatchCompletion(); // wait for the previous batch to complete, if there is one
        if (batchCount != 0) {
            // If we have any accumulated reads that haven't been dispatched, start one last
            // async batch write, and then wait for it to complete
            startAsynchronousBatchWrite();
            waitForPreviousBatchCompletion();
        }

        return true;
    }

    private void startAsynchronousBatchWrite() {
        // Send an ASYNCHRONOUS command to Python to tell it to start consuming the lines about to be written
        // to the FIFO through the stream writer. Sending a *SYNCHRONOUS* command would block indefinitely
        // waiting for the ack to be received, but it never would be because no data has been written to the
        // stream yet.
        pythonExecutor.sendAsynchronousCommand(
                String.format("for i in range(%s):\n    tempFile.write(tool.readDataFIFO())" + NL + NL, batchCount)
        );
        asyncStreamWriter.startBatchWrite(batchList);
        batchList = new ArrayList<>(batchSize);
        batchCount = 0;
    }

    private void waitForPreviousBatchCompletion() {
        if (asyncStreamWriter.waitForPreviousBatchCompletion() != null) {
            // The writing has completed, now also wait for the Python code to send the ack indicating it
            // has consumed all of the data from the stream.
            pythonExecutor.waitForAck(); // block waiting for the ack..
        }
    }

    @Override
    public void closeTool() {
        // Send a synchronous command to Python to close the temp file
        pythonExecutor.sendSynchronousCommand("tempFile.close()" + NL);

        // terminate the async writer and fifo
        pythonExecutor.terminate();
    }
}
