package org.broadinstitute.hellbender.tools.spark.sv.evidence;

import com.google.common.annotations.VisibleForTesting;
import htsjdk.samtools.*;
import htsjdk.samtools.util.SequenceUtil;
import org.broadinstitute.hellbender.exceptions.GATKException;
import org.broadinstitute.hellbender.tools.spark.sv.utils.*;
import org.broadinstitute.hellbender.tools.spark.sv.utils.SVFastqUtils.FastqRead;
import org.broadinstitute.hellbender.tools.spark.utils.HopscotchMultiMap;
import org.broadinstitute.hellbender.utils.BaseUtils;
import org.broadinstitute.hellbender.utils.bwa.*;
import org.broadinstitute.hellbender.utils.fermi.FermiLiteAssembler;
import org.broadinstitute.hellbender.utils.fermi.FermiLiteAssembly;
import org.broadinstitute.hellbender.utils.fermi.FermiLiteAssembly.Contig;
import org.broadinstitute.hellbender.utils.fermi.FermiLiteAssembly.Connection;
import org.broadinstitute.hellbender.utils.gcs.BucketUtils;
import org.broadinstitute.hellbender.utils.io.IOUtils;
import scala.Tuple2;

import java.io.*;
import java.util.*;

/** LocalAssemblyHandler that uses FermiLite. */
public final class FermiLiteAssemblyHandler implements FindBreakpointEvidenceSpark.LocalAssemblyHandler {
    private static final long serialVersionUID = 1L;
    private static final int assemblyKmerSize = 31;
    private final String alignerIndexFile;
    private final int maxFastqSize;
    private final String fastqDir;
    private final boolean writeGFAs;
    private final boolean popVariantBubbles;
    private final boolean removeShadowedContigs;
    private final boolean expandAssemblyGraph;
    private final int zDropoff;

    public FermiLiteAssemblyHandler( final String alignerIndexFile, final int maxFastqSize,
                                     final String fastqDir, final boolean writeGFAs,
                                     final boolean popVariantBubbles, final boolean removeShadowedContigs,
                                     final boolean expandAssemblyGraph, final int zDropoff ) {
        this.alignerIndexFile = alignerIndexFile;
        this.maxFastqSize = maxFastqSize;
        this.fastqDir = fastqDir;
        this.writeGFAs = writeGFAs;
        this.popVariantBubbles = popVariantBubbles;
        this.removeShadowedContigs = removeShadowedContigs;
        this.expandAssemblyGraph = expandAssemblyGraph;
        this.zDropoff = zDropoff;
    }

    /** This method creates an assembly with FermiLite, and uses the graph information returned by that
     *  assembler to stitch together valid paths through the contigs.
     *  These paths are then aligned to reference with BWA. */
    @Override
    public AlignedAssemblyOrExcuse apply( final Tuple2<Integer, List<FastqRead>> intervalAndReads ) {
        final int intervalID = intervalAndReads._1();
        final String assemblyName = AlignedAssemblyOrExcuse.formatAssemblyID(intervalID);
        final List<FastqRead> readsList = intervalAndReads._2();

        // bail if the assembly will be too large
        final int fastqSize = readsList.stream().mapToInt(fastqRead -> fastqRead.getBases().length).sum();
        if ( fastqSize > maxFastqSize ) {
            return new AlignedAssemblyOrExcuse(intervalID, "no assembly -- too big (" + fastqSize + " bytes).");
        }

        // record the reads in the assembly as a FASTQ, if requested
        if ( fastqDir != null ) {
            final String fastqName = String.format("%s/%s.fastq", fastqDir, assemblyName);
            final ArrayList<FastqRead> sortedReads = new ArrayList<>(readsList);
            sortedReads.sort(Comparator.comparing(FastqRead::getHeader));
            SVFastqUtils.writeFastqFile(fastqName, sortedReads.iterator());
        }

        // assemble the reads
        final FermiLiteAssembler assembler = new FermiLiteAssembler();
        if ( popVariantBubbles ) {
            final int MAG_F_AGGRESSIVE = 0x20; // pop variant bubbles (not default)
            final int MAG_F_POPOPEN = 0x40; // aggressive tip trimming (default)
            final int MAG_F_NO_SIMPL = 0x80; // skip bubble simplification (default)
            // add aggressive-popping flag, and remove no-simplification flag
            assembler.setCleaningFlag(MAG_F_AGGRESSIVE | MAG_F_POPOPEN);
        }
        final long timeStart = System.currentTimeMillis();
        final FermiLiteAssembly initialAssembly = assembler.createAssembly(readsList);
        final int secondsInAssembly = (int)((System.currentTimeMillis() - timeStart + 500)/1000);
        if ( initialAssembly.getNContigs() == 0 ) {
            return new AlignedAssemblyOrExcuse(intervalID, "no assembly -- no contigs produced by assembler.");
        }

        // patch up the assembly to improve contiguity
        final FermiLiteAssembly assembly = reviseAssembly(initialAssembly, removeShadowedContigs, expandAssemblyGraph);

        final float assemblyScore = scoreAssembly(readsList, assembly, assemblyName, fastqDir);

        final int readBases = readsList.stream().mapToInt(read -> read.getBases().length).sum();

        // record the assembly as a GFA, if requested
        if ( fastqDir != null && writeGFAs ) {
            final String gfaName =  String.format("%s/%s.gfa", fastqDir, assemblyName);
            try ( final Writer writer = new BufferedWriter(new OutputStreamWriter(BucketUtils.createFile(gfaName))) ) {
                assembly.writeGFA(writer);
            }
            catch ( final IOException ioe ) {
                throw new GATKException("Can't write "+gfaName, ioe);
            }
        }

        // align the assembled contigs to the genomic reference
        try ( final BwaMemAligner aligner = new BwaMemAligner(BwaMemIndexCache.getInstance(alignerIndexFile)) ) {
            aligner.setIntraCtgOptions();
            aligner.setZDropOption(zDropoff);
            final List<byte[]> sequences =
                    assembly.getContigs().stream()
                            .map(Contig::getSequence)
                            .collect(SVUtils.arrayListCollector(assembly.getNContigs()));
            final List<List<BwaMemAlignment>> alignments = aligner.alignSeqs(sequences);
            return new AlignedAssemblyOrExcuse(intervalID, assembly, secondsInAssembly, assemblyScore, readBases, alignments);
        }
    }

    @VisibleForTesting
    static FermiLiteAssembly reviseAssembly( final FermiLiteAssembly initialAssembly,
                                             final boolean removeShadowedContigs,
                                             final boolean expandAssemblyGraph ) {
        final FermiLiteAssembly unshadowedAssembly =
                removeShadowedContigs ? removeShadowedContigs(initialAssembly) : initialAssembly;
        final FermiLiteAssembly expandedAssembly =
                expandAssemblyGraph ? expandAssemblyGraph(removeUnbranchedConnections(unshadowedAssembly)) : unshadowedAssembly;
        return removeShadowedContigs ? removeShadowedContigs(expandedAssembly) : expandedAssembly;
    }

    /** Eliminate contigs from the assembly that vary from another contig by just a few SNVs. */
    @VisibleForTesting
    static FermiLiteAssembly removeShadowedContigs( final FermiLiteAssembly assembly ) {
        final double maxMismatchRate = .01;

        // make a map of assembled kmers
        final HopscotchMultiMap<SVKmerShort, ContigLocation, KmerLocation> kmerMap = kmerizeAssembly(assembly);

        /* Remove contigs that vary by a small number of SNVs from another contig. E.g.,
         *  <-------contigA----------->
         *     ||x|||||x||||x|||
         *     <---contigB----->
         * ContigB is "shadowed" by contigA because it provides a complete alignment for all of contigB that contains
         * just a few mismatches.
         */
        final Set<Contig> contigsToRemove = new HashSet<>();
        assembly.getContigs().forEach(tig -> {
            final Set<ContigLocation> testedLocations = new HashSet<>();
            final byte[] tigBases = tig.getSequence();
            final int maxMismatches = (int)(tigBases.length * maxMismatchRate);
            int tigOffset = 0;
            final SVKmerizer contigKmerItr =
                    new SVKmerizer(tig.getSequence(), assemblyKmerSize, new SVKmerShort(assemblyKmerSize));
            while ( contigKmerItr.hasNext() ) {
                final SVKmerShort contigKmer = (SVKmerShort)contigKmerItr.next();
                final SVKmerShort canonicalContigKmer = contigKmer.canonical(assemblyKmerSize);
                final boolean contigKmerIsCanonical = contigKmer.equals(canonicalContigKmer);
                final Iterator<KmerLocation> locItr = kmerMap.findEach(canonicalContigKmer);
                while ( locItr.hasNext() ) {
                    final ContigLocation tig2Location = locItr.next().getLocation();
                    final Contig tig2 = tig2Location.getContig();
                    if ( tig == tig2 || contigsToRemove.contains(tig2) ) continue;
                    //TODO: rewrite some of these calculations using ContigLocation methods rc(), upstream(), etc.
                    // having found a kmer that matches between two contigs, and knowing the offsets of those kmers, we'll
                    // figure out the regions of the two contigs that overlap if the shared kmer implies a valid identity
                    final byte[] tig2Bases = tig2.getSequence();
                    final boolean isRC = contigKmerIsCanonical != tig2Location.isCanonical();
                    final int tig2Offset =
                            isRC ? tig2Bases.length - tig2Location.getOffset() - assemblyKmerSize : tig2Location.getOffset();
                    // if the number of bases upstream of the matching kmer is greater for tig than for tig2, then
                    // tig2 doesn't completely cover tig and so can't shadow it
                    if ( tigOffset > tig2Offset ) continue;
                    // similarly for the bases downstream of the matching kmer.  if tig has more of them than tig2,
                    // then tig isn't shadowed by tig2.
                    if ( tigBases.length - tigOffset > tig2Bases.length - tig2Offset ) continue;
                    final int tig2Start = tig2Offset - tigOffset;
                    // in a lengthy region of complete identity many kmers will imply the same alignment --
                    // test each alignment just once
                    if ( testedLocations.add(new ContigLocation(tig2, tig2Start, isRC)) ) {
                        int nMismatches = 0;
                        if ( !isRC ) {
                            for ( int idx = 0; idx != tigBases.length; ++idx ) {
                                if ( tigBases[idx] != tig2Bases[tig2Start+idx] ) {
                                    nMismatches += 1;
                                    if ( nMismatches > maxMismatches ) break;
                                }
                            }
                        } else {
                            final int tig2RCOffset = tig2Bases.length - tig2Start - 1;
                            for ( int idx = 0; idx != tigBases.length; ++idx ) {
                                if ( tigBases[idx] != BaseUtils.simpleComplement(tig2Bases[tig2RCOffset-idx]) ) {
                                    nMismatches += 1;
                                    if ( nMismatches > maxMismatches ) break;
                                }
                            }
                        }
                        if ( nMismatches <= maxMismatches ) {
                            contigsToRemove.add(tig);
                            break;
                        }
                    }
                }
                tigOffset += 1;
            }
        });

        // make a new contig list without the shadowed contigs
        final List<Contig> contigList = new ArrayList<>(assembly.getContigs().size()-contigsToRemove.size());
        assembly.getContigs().stream()
                .filter(tig -> !contigsToRemove.contains(tig))
                .forEach(contigList::add);

        // note which surviving contigs have connections to deleted contigs
        final Set<Contig> staleConnectionContigs = new HashSet<>(SVUtils.hashMapCapacity(contigsToRemove.size()));
        contigsToRemove.forEach(tig ->
                tig.getConnections().forEach(conn ->
                        staleConnectionContigs.add(conn.getTarget())));

        // remove the connections that refer to deleted contigs
        staleConnectionContigs.forEach(tig -> {
            final List<Connection> connections = new ArrayList<>(tig.getConnections().size() - 1);
            tig.getConnections().stream()
                    .filter(conn -> !contigsToRemove.contains(conn.getTarget()))
                    .forEach(connections::add);
            tig.setConnections(connections);
        });

        return new FermiLiteAssembly(contigList);
    }

    /** make a map of assembled kmers */
    private static HopscotchMultiMap<SVKmerShort, ContigLocation, KmerLocation> kmerizeAssembly(
                    final FermiLiteAssembly assembly ) {
        final int capacity =
                assembly.getContigs().stream().mapToInt(tig -> tig.getSequence().length - assemblyKmerSize + 1).sum();
        final HopscotchMultiMap<SVKmerShort, ContigLocation, KmerLocation> kmerMap = new HopscotchMultiMap<>(capacity);
        assembly.getContigs().forEach(tig -> {
            int contigOffset = 0;
            final Iterator<SVKmer> contigKmerItr =
                    new SVKmerizer(tig.getSequence(), assemblyKmerSize, new SVKmerShort());
            while ( contigKmerItr.hasNext() ) {
                final SVKmerShort kmer = (SVKmerShort)contigKmerItr.next();
                final SVKmerShort canonicalKmer = kmer.canonical(assemblyKmerSize);
                final ContigLocation location = new ContigLocation(tig, contigOffset++, kmer.equals(canonicalKmer));
                kmerMap.add(new KmerLocation(canonicalKmer, location));
            }
        });
        return kmerMap;
    }

    // join contigs that connect without any other branches
    // i.e., if contig A has contig B as its sole successor, and contig B has contig A as its sole predecessor,
    // then combine contigs A and B into a single contig AB.
    @VisibleForTesting
    static FermiLiteAssembly removeUnbranchedConnections( final FermiLiteAssembly assembly ) {
        final int nContigs = assembly.getNContigs();
        final List<Contig> contigList = new ArrayList<>(nContigs);
        final Set<Contig> examined = new HashSet<>(SVUtils.hashMapCapacity(nContigs));

        // find contigs with a single predecessor contig that has a single successor (or vice versa) and join them
        assembly.getContigs().forEach(tig -> {
            if ( !examined.add(tig) ) return;
            Connection conn;
            Connection conn2;
            while ( (conn = tig.getSolePredecessor()) != null &&
                    !examined.contains(conn.getTarget()) &&
                    (conn2 = conn.getTarget().getSingletonConnection(!conn.isTargetRC())) != null ) {
                examined.add(conn.getTarget());
                tig = joinContigsWithConnections(tig, conn, conn2);
            }
            while ( (conn = tig.getSoleSuccessor()) != null &&
                    !examined.contains(conn.getTarget()) &&
                    (conn2 = conn.getTarget().getSingletonConnection(!conn.isTargetRC())) != null ) {
                examined.add(conn.getTarget());
                tig = joinContigsWithConnections(tig, conn, conn2);
            }
            contigList.add(tig);
        });
        return new FermiLiteAssembly(contigList);
    }

    // combine two connected contigs into one, preserving all their connections, except their connections to each other.
    // (i.e., firstContig gets joined to connection.getTarget().)  the connection argument is the pointer from
    // firstContig to the 2nd contig, and rcConnection is the back-pointer from 2nd contig to firstContig.
    private static Contig joinContigsWithConnections( final Contig firstContig,
                                                      final Connection connection,
                                                      final Connection rcConnection ) {
        final Contig joinedContig = joinContigs(firstContig, Collections.singletonList(connection));
        final Contig lastContig = connection.getTarget();
        final int capacity = firstContig.getConnections().size() + lastContig.getConnections().size() - 2;
        final List<Connection> connections = new ArrayList<>(capacity);
        for ( final Connection conn : firstContig.getConnections() ) {
            if ( conn != connection ) {
                final Connection newConnection =
                        new Connection(conn.getTarget(), conn.getOverlapLen(), true, conn.isTargetRC());
                replaceConnection(conn.getTarget(),
                                    conn.rcConnection(firstContig),
                                    newConnection.rcConnection(joinedContig));
                connections.add(newConnection);
            }
        }
        for ( final Connection conn : lastContig.getConnections() ) {
            if ( conn != rcConnection ) {
                final Connection newConnection =
                        new Connection(conn.getTarget(), conn.getOverlapLen(), false, conn.isTargetRC());
                replaceConnection(conn.getTarget(),
                                    conn.rcConnection(lastContig),
                                    newConnection.rcConnection(joinedContig));
                connections.add(newConnection);
            }
        }
        joinedContig.setConnections(connections);
        return joinedContig;
    }

    private static void replaceConnection( final Contig contig,
                                           final Connection oldConnection,
                                           final Connection newConnection ) {
        final List<Connection> oldConnections = contig.getConnections();
        final List<Connection> newConnections = new ArrayList<>(oldConnections.size());
        for ( final Connection conn : oldConnections ) {
            final Connection toAdd;
            if ( conn.getTarget() == oldConnection.getTarget() &&
                    conn.isRC() == oldConnection.isRC() &&
                    conn.isTargetRC() == oldConnection.isTargetRC() ) {
                toAdd = newConnection;
            } else {
                toAdd = conn;
            }
            newConnections.add(toAdd);
        }
        contig.setConnections(newConnections);
    }

    // join the sequences of a chain of contigs to produce a single, new contig
    private static Contig joinContigs( final Contig firstContig, final List<Connection> path ) {
        if ( path.isEmpty() ) return firstContig;
        final int nSupportingReads =
                path.stream()
                        .mapToInt(conn -> conn.getTarget().getNSupportingReads())
                        .reduce(firstContig.getNSupportingReads(), Integer::sum);
        final int newContigLen =
                path.stream()
                        .mapToInt(conn -> conn.getTarget().getSequence().length - conn.getOverlapLen())
                        .reduce(firstContig.getSequence().length, Integer::sum);
        final byte[] sequence = new byte[newContigLen];
        int destinationOffset = firstContig.getSequence().length;
        System.arraycopy(firstContig.getSequence(), 0, sequence, 0, destinationOffset);
        if ( path.get(0).isRC() )
            SequenceUtil.reverseComplement(sequence, 0, destinationOffset);
        for ( final Connection conn : path ) {
            final byte[] contigSequence = conn.getTarget().getSequence();
            final int len = contigSequence.length - conn.getOverlapLen();
            if ( !conn.isTargetRC() ) {
                System.arraycopy(contigSequence, conn.getOverlapLen(), sequence, destinationOffset, len);
            } else {
                System.arraycopy(contigSequence, 0, sequence, destinationOffset, len);
                SequenceUtil.reverseComplement(sequence, destinationOffset, len);
            }
            destinationOffset += len;
        }
        return new Contig(sequence, null, nSupportingReads);
    }

    // walk the graph to trace out all paths, breaking only at cycles and at contigs that require phasing
    // N.B.: the per-base-coverage info is stripped away when contigs are joined.
    @VisibleForTesting
    static FermiLiteAssembly expandAssemblyGraph( final FermiLiteAssembly assembly ) {
        final int nContigs = assembly.getNContigs();
        // this will hold the new compound contigs that we build
        final List<Contig> contigList = new ArrayList<>(nContigs);

        // this will keep track of where we've been during a depth-first exploration from some starting contig.
        // we'll use it to kill cycles.
        final Set<ContigStrand> visited = new HashSet<>();

        // this will keep track of contigs that we've touched during any of our explorations.
        // we'll use it to avoid backward tracing from contigs that have already appeared on some path.
        final Set<Contig> examined = new HashSet<>(SVUtils.hashMapCapacity(nContigs));

        // trace paths from sources and sinks
        assembly.getContigs().forEach(tig -> {
            if ( examined.contains(tig) ) return;
            if ( tig.getConnections().isEmpty() ) {
                contigList.add(tig);
                examined.add(tig);
            } else {
                final int nPredecessors = countPredecessors(tig);
                final int nSuccessors = tig.getConnections().size() - nPredecessors;
                if ( nPredecessors == 0 ) { // found a source
                    tracePaths(tig, false, contigList, examined, visited);
                } else if ( nSuccessors == 0 ) { // found a sink
                    tracePaths(tig, true, contigList, examined, visited);
                }
            }
        });

        // once you've traced paths from all sources and sinks you should be done, right?
        // not so fast, there!  how about A -> B -> C -> A, i.e., a smooth, circular cycle?
        // anything not examined must be part of a smooth cycle -- just start anywhere to pick it up
        assembly.getContigs().forEach(tig -> {
            if ( !examined.contains(tig) ) {
                tracePaths(tig, false, contigList, examined, visited);
            }
        });

        return new FermiLiteAssembly(contigList);
    }

    private static int countPredecessors( final Contig contig ) {
        return contig.getConnections().stream().mapToInt(conn -> conn.isRC() ? 1 : 0).sum();
    }

    // called for each connected source and each connected sink that hasn't already been examined.
    // starts a depth-first search through the graph.
    private static void tracePaths( final Contig contig,
                                    final boolean isRC,
                                    final List<Contig> contigList,
                                    final Set<Contig> examined,
                                    final Set<ContigStrand> visited ) {
        examined.add(contig);
        final LinkedList<Connection> path = new LinkedList<>();
        final ContigStrand contigStrand = new ContigStrand(contig, isRC);
        final boolean isCycle = !visited.add(contigStrand);
        for ( Connection connection : contig.getConnections() ) {
            if ( connection.isRC() == isRC && connection.getOverlapLen() >= 0 ) {
                extendPath(contig, connection, path, contigList, examined, visited);
            }
        }
        if ( !isCycle ) visited.remove(contigStrand);
    }

    // called to add another connection to the path we're building.
    private static void extendPath( final Contig firstContig,
                                    final Connection connection,
                                    final LinkedList<Connection> path,
                                    final List<Contig> contigList,
                                    final Set<Contig> examined,
                                    final Set<ContigStrand> visited ) {
        // note that the connection gets added to the path regardless of whether it's cyclic or not.
        // this will create a contig with exactly one repeat of the first contig that we detect to be part of a cycle.
        // so, for example, A -> B -> C -> C will produce a contig ABCC, but not ABCCC or ABCCCC, etc.
        path.addLast(connection);
        final ContigStrand contigStrand = new ContigStrand(connection.getTarget(), connection.isTargetRC());
        final boolean isCycle = !visited.add(contigStrand);
        boolean atEndOfPath = true;
        if ( !isCycle ) {
            final Contig target = connection.getTarget();
            final int nPredecessors = countPredecessors(target);
            final int nSuccessors = target.getConnections().size() - nPredecessors;
            final boolean needsPhasing = nPredecessors > 1 && nSuccessors > 1;
            if ( needsPhasing ) {
                // the first time we find a contig that needs phasing info to avoid creating false joins:
                // we end the current path at that contig, but initiate a new path from that contig.
                // i.e., we pretend that it's both a sink and a source, even though it's neither.
                // e.g., if the assembly has the structure A->C, B->C, C->D, C->E, then C has two predecessors (A and B)
                // and two successors (D and E).  i.e., it needs phasing.
                // the correct paths are probably either ACD + BCE or ACE + BCD, but we can't tell which without
                // phasing info.  so we're just going to produce AC, BC, CD, and CE so that we don't make up lies.
                if ( examined.add(target) ) {
                    tracePaths(target, connection.isTargetRC(), contigList, examined, visited);
                }
            } else {
                examined.add(target);

                for ( Connection conn : target.getConnections() ) {
                    if ( conn.isRC() == connection.isTargetRC() && conn.getOverlapLen() >= 0 ) {
                        extendPath(firstContig, conn, path, contigList, examined, visited);
                        atEndOfPath = false;
                    }
                }
            }
        }
        if ( atEndOfPath ) {
            contigList.add(joinContigs(firstContig, path));
        }
        if ( !isCycle ) visited.remove(contigStrand);
        path.removeLast();
    }

    // the value class associated with maps of kmers from a contig.
    // which contig, where in the contig, and whether the kmer is or isn't canonical in the contig sequence as written.
    private static final class ContigLocation {
        private final Contig contig;
        private final int offset;
        private final boolean canonical;

        public ContigLocation(final Contig contig, final int offset, final boolean canonical ) {
            this.contig = contig;
            this.offset = offset;
            this.canonical = canonical;
        }
        public Contig getContig() { return contig; }
        public int getOffset() { return offset; }
        public boolean isCanonical() { return canonical; }

        public ContigLocation upstream( final int distance ) {
            return new ContigLocation(contig, offset - distance, canonical);
        }

        public ContigLocation rc() {
            return new ContigLocation(contig, contig.getSequence().length - offset - 1, !canonical);
        }

        @Override public boolean equals( final Object obj ) {
            return obj instanceof ContigLocation && equals((ContigLocation)obj);
        }

        public boolean equals( final ContigLocation that ) {
            if ( this == that ) return true;
            return contig == that.contig && offset == that.offset && canonical == that.canonical;
        }

        @Override public int hashCode() {
            return 47 * (47 * (contig.hashCode() + 47 * offset) + (canonical ? 31 : 5));
        }
    }

    // map entry for maps of kmers onto contig locations
    private static final class KmerLocation implements Map.Entry<SVKmerShort, ContigLocation> {
        private final SVKmerShort kmer;
        private final ContigLocation location;

        public KmerLocation( final SVKmerShort kmer, final ContigLocation location ) {
            this.kmer = kmer;
            this.location = location;
        }

        public SVKmerShort getKmer() { return kmer; }
        public ContigLocation getLocation() { return location; }
        @Override public SVKmerShort getKey() { return kmer; }
        @Override public ContigLocation getValue() { return location; }
        @Override public ContigLocation setValue(final ContigLocation value ) {
            throw new UnsupportedOperationException("KmerLocation is immutable");
        }
    }

    // contig + strand info.
    private static final class ContigStrand {
        private final Contig contig;
        private final boolean isRC;

        public ContigStrand( final Contig contig, final boolean isRC ) {
            this.contig = contig;
            this.isRC = isRC;
        }

        public Contig getContig() { return contig; }
        public boolean isRC() { return isRC; }

        public ContigStrand rc() { return new ContigStrand(contig, !isRC); }

        @Override public boolean equals( final Object obj ) {
            return obj instanceof ContigStrand && equals((ContigStrand) obj);
        }

        public boolean equals( final ContigStrand that ) {
            if ( this == that ) return true;
            return contig == that.contig && isRC == that.isRC;
        }

        @Override public int hashCode() {
            return isRC ? -contig.hashCode() : contig.hashCode();
        }
    }

    private static float scoreAssembly( final List<FastqRead> readsList,
                                        final FermiLiteAssembly assembly,
                                        final String assemblyName,
                                        final String fastqDir ) {
        final String fastaFile = String.format("%s/%s.fasta", fastqDir, assemblyName);
        final String imageFile;
        final List<Contig> contigs = assembly.getContigs();
        final int nContigs = contigs.size();
        try {
            final File tmpFasta = File.createTempFile(assemblyName, ".fasta");
            final String tmpFastaFile = tmpFasta.getPath();
            imageFile = tmpFastaFile + ".img";
            try ( final BufferedWriter writer1 =
                          new BufferedWriter(new OutputStreamWriter(BucketUtils.createFile(fastaFile)));
                  final BufferedWriter writer2 =
                          new BufferedWriter(new OutputStreamWriter(BucketUtils.createFile(tmpFastaFile)))) {
                for ( int idx = 0; idx != nContigs; ++idx ) {
                    final String header = String.format(">tig%05d", idx);
                    final String sequence = new String(contigs.get(idx).getSequence());
                    writer1.write(header);
                    writer1.newLine();
                    writer1.write(sequence);
                    writer1.newLine();
                    writer2.write(header);
                    writer2.newLine();
                    writer2.write(sequence);
                    writer2.newLine();
                }
            }
            BwaMemIndex.createIndexImageFromFastaFile(tmpFastaFile, imageFile);
            tmpFasta.delete();
        } catch ( final IOException ioe ) {
            throw new GATKException("can't write local assembly as fasta file", ioe);
        }

        final List<List<BwaMemAlignment>> alignments;
        try ( final BwaMemIndex index = new BwaMemIndex(imageFile) ) {
            final BwaMemAligner aligner = new BwaMemAligner(index);
            aligner.setIntraCtgOptions();
            aligner.setZDropOption(20);
            aligner.alignPairs();
            alignments = aligner.alignSeqs(readsList, FastqRead::getBases);
        }
        new File(imageFile).delete();

        final List<SAMSequenceRecord> recList = new ArrayList<>(nContigs);
        final List<String> refNames = new ArrayList<>(nContigs);
        for ( int idx = 0; idx != nContigs; ++idx ) {
            final String contigName = String.format("tig%05d", idx);
            recList.add(new SAMSequenceRecord(contigName, contigs.get(idx).getSequence().length));
            refNames.add(contigName);
        }
        final SAMFileHeader header = new SAMFileHeader(new SAMSequenceDictionary(recList));
        header.setSortOrder(SAMFileHeader.SortOrder.coordinate);
        final List<SAMRecord> samReads =
                new ArrayList<>(alignments.stream().mapToInt(alnList -> Math.min(1,alnList.size())).sum());
        final int nReads = readsList.size();
        for ( int idx = 0; idx != nReads; ++idx ) {
            final FastqRead read = readsList.get(idx);
            final List<BwaMemAlignment> readAlignments = alignments.get(idx);
            if ( readAlignments.isEmpty() ) {
                final SAMRecord unmappedRec = new SAMRecord(header);
                unmappedRec.setReadName(read.getName());
                unmappedRec.setReadBases(read.getBases());
                unmappedRec.setBaseQualities(read.getQuals());
                samReads.add(unmappedRec);
                continue;
            }
            BwaMemAlignmentUtils.toSAMStreamForRead(read.getName(), read.getBases(), read.getQuals(), readAlignments,
                    header, refNames, null).forEach(samReads::add);
        }
        samReads.sort(new SAMRecordCoordinateComparator());
        final String bamFile = String.format("%s/%s.bam", fastqDir, assemblyName);
        SVFileUtils.writeSAMFile(bamFile, samReads.iterator(), header, true);
        return (float)samReads.stream().mapToInt(FermiLiteAssemblyHandler::assemblyScore).sum()/samReads.size();
    }

    private static int assemblyScore( final SAMRecord read ) {
        final Integer score = read.getIntegerAttribute("AS");
        return score == null ? 0 : score;
    }

    private List<List<UngappedAlignment>> alignReads( final FermiLiteAssembly assembly, final List<FastqRead> reads ) {
        HopscotchMultiMap<SVKmerShort, ContigLocation, KmerLocation> kmerMap = kmerizeAssembly(assembly);
        final List<List<UngappedAlignment>> allAlignments = new ArrayList<>(reads.size());
        for ( final FastqRead read : reads ) {
            final List<UngappedAlignment> alignments = new ArrayList<>();
            allAlignments.add(alignments);
            final Map<ContigLocation, UngappedAlignment> alignmentMap = new HashMap<>();
            int readOffset = 0;
            SVKmerizer readKmerIter =
                    new SVKmerizer(read.getBases(), assemblyKmerSize, new SVKmerShort(assemblyKmerSize));
            while ( readKmerIter.hasNext() ) {
                final SVKmerShort readKmer = (SVKmerShort)readKmerIter.next();
                final SVKmerShort canonicalReadKmer = readKmer.canonical(assemblyKmerSize);
                final boolean readKmerIsCanonical = readKmer.equals(canonicalReadKmer);
                final Iterator<KmerLocation> kmerLocationIterator = kmerMap.findEach(canonicalReadKmer);
                while ( kmerLocationIterator.hasNext() ) {
                    final KmerLocation kmerLocation = kmerLocationIterator.next();
                    final ContigLocation contigLocation = kmerLocation.getLocation();
                    final Contig contig = contigLocation.getContig();
                    // the contig location corresponding to the start of the read -- NB: may have a negative offset
                    final ContigLocation startingLocation =
                        readKmerIsCanonical == contigLocation.isCanonical() ?
                                contigLocation.upstream(readOffset) :
                                contigLocation.rc().upstream(readOffset + assemblyKmerSize - 1);
                    if ( alignmentMap.containsKey(startingLocation) ) continue;
                    final int unmatchedQualityPenalty = scoreAlignment(read, startingLocation);
                    final UngappedAlignment ungappedAlignment;
                    if ( startingLocation.getOffset() < 0 ) {
                        final ContigLocation adjustedLocation =
                                new ContigLocation(contig, 0, startingLocation.isCanonical());
                        final int readStart = -startingLocation.getOffset();
                        ungappedAlignment =
                                new UngappedAlignment(adjustedLocation,
                                            readStart,
                                            Math.min(read.getBases().length - readStart, contig.getSequence().length),
                                            unmatchedQualityPenalty);
                    } else {
                        final int alignmentLength =
                                Math.min(read.getBases().length, contig.getSequence().length - startingLocation.getOffset());
                        ungappedAlignment =
                                new UngappedAlignment(startingLocation, 0, alignmentLength, unmatchedQualityPenalty);
                    }
                    alignmentMap.put(startingLocation, ungappedAlignment);
                    readOffset += 1;
                }
            }
            final int minPenalty =
                    alignmentMap.values().stream()
                            .mapToInt(UngappedAlignment::getUnmatchedQualityPenalty).min().orElse(0);
            alignmentMap.values().stream()
                    .filter(aln -> aln.getUnmatchedQualityPenalty() == minPenalty).forEach(alignments::add);
        }
        return allAlignments;
    }

    private static int scoreAlignment( final FastqRead read, final ContigLocation startingLocation ) {
        final byte[] readBases = read.getBases();
        final byte[] readQuals = read.getQuals();
        final byte[] contigBases = startingLocation.getContig().getSequence();
        int unmatchedQualitySum = 0;
        if ( startingLocation.isCanonical() ) {
            int contigIndex = startingLocation.getOffset();
            for ( int readIndex = 0; readIndex != readBases.length; ++readIndex, ++contigIndex ) {
                if ( contigIndex < 0 || contigIndex >= contigBases.length ||
                        readBases[readIndex] != contigBases[contigIndex] ) {
                    unmatchedQualitySum += readQuals[readIndex];
                }
            }
        } else {
            int contigIndex = contigBases.length - startingLocation.getOffset() - 1;
            for ( int readIndex = 0; readIndex != readBases.length; ++readIndex, --contigIndex ) {
                if ( contigIndex < 0 || contigIndex >= contigBases.length ||
                        readBases[readIndex] != BaseUtils.simpleComplement(contigBases[contigIndex]) ) {
                    unmatchedQualitySum += readQuals[readIndex];
                }
            }
        }
        return unmatchedQualitySum;
    }

    private static final class UngappedAlignment {
        private final ContigLocation contigLocation;
        private final int readOffset;
        private final int alignedLength;
        private final int unmatchedQualityPenalty;

        public UngappedAlignment( final ContigLocation contigLocation,
                                  final int readOffset, final int alignedLength,
                                  final int unmatchedQualityPenalty) {
            this.contigLocation = contigLocation;
            this.readOffset = readOffset;
            this.alignedLength = alignedLength;
            this.unmatchedQualityPenalty = unmatchedQualityPenalty;
        }

        public ContigLocation getContigLocation() { return contigLocation; }
        public int getReadOffset() { return readOffset; }
        public int getAlignedLength() { return alignedLength; }
        public int getUnmatchedQualityPenalty() { return unmatchedQualityPenalty; }
    }
}
