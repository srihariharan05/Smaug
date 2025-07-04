/*
 * Copyright (c) 2014-2015 ARM Limited
 * All rights reserved
 *
 * The license below extends only to copyright in the software and shall
 * not be construed as granting a license to any other intellectual
 * property including but not limited to intellectual property relating
 * to a hardware implementation of the functionality of the software
 * licensed hereunder.  You may use the software subject to the license
 * terms below provided that you ensure that this notice is replicated
 * unmodified and in its entirety in all distributions of the software,
 * modified or unmodified, in source code or in binary form.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Authors: Mitch Hayenga
 */

#ifndef __MEM_CACHE_PREFETCH_QUEUED_HH__
#define __MEM_CACHE_PREFETCH_QUEUED_HH__

#include <cstdint>
#include <list>
#include <utility>

#include "base/statistics.hh"
#include "base/types.hh"
#include "mem/cache/prefetch/base.hh"
#include "mem/packet.hh"

struct QueuedPrefetcherParams;

class QueuedPrefetcher : public BasePrefetcher
{
  protected:
    struct DeferredPacket : public BaseTLB::Translation {
        /** Owner of the packet */
        QueuedPrefetcher *owner;
        /** Prefetch info corresponding to this packet */
        PrefetchInfo pfInfo;
        /** Time when this prefetch becomes ready */
        Tick tick;
        /** The memory packet generated by this prefetch */
        PacketPtr pkt;
        /** The priority of this prefetch */
        int32_t priority;
        /** Request used when a translation is needed */
        RequestPtr translationRequest;
        ThreadContext *tc;
        bool ongoingTranslation;

        /**
         * Constructor
         * @param o QueuedPrefetcher in charge of this request
         * @param pfi PrefechInfo object associated to this packet
         * @param t Time when this prefetch becomes ready
         * @param p PacketPtr with the memory request of the prefetch
         * @param prio This prefetch priority
         */
        DeferredPacket(QueuedPrefetcher *o, PrefetchInfo const &pfi, Tick t,
            int32_t prio) : owner(o), pfInfo(pfi), tick(t), pkt(nullptr),
            priority(prio), translationRequest(), tc(nullptr),
            ongoingTranslation(false) {
        }

        bool operator>(const DeferredPacket& that) const
        {
            return priority > that.priority;
        }
        bool operator<(const DeferredPacket& that) const
        {
            return priority < that.priority;
        }
        bool operator<=(const DeferredPacket& that) const
        {
            return !(*this > that);
        }

        /**
         * Create the associated memory packet
         * @param paddr physical address of this packet
         * @param blk_size block size used by the prefetcher
         * @param mid Requester ID of the access that generated this prefetch
         * @param tag_prefetch flag to indicate if the packet needs to be
         *        tagged
         * @param t time when the prefetch becomes ready
         */
        void createPkt(Addr paddr, unsigned blk_size, MasterID mid,
                       bool tag_prefetch, Tick t);

        /**
         * Sets the translation request needed to obtain the physical address
         * of this request.
         * @param req The Request with the virtual address of this request
         */
        void setTranslationRequest(const RequestPtr &req)
        {
            translationRequest = req;
        }

        void markDelayed() override
        {}

        void finish(const Fault &fault, const RequestPtr &req,
                            ThreadContext *tc, BaseTLB::Mode mode) override;

        /**
         * Issues the translation request to the provided TLB
         * @param tlb the tlb that has to translate the address
         */
        void startTranslation(BaseTLB *tlb);
    };

    std::list<DeferredPacket> pfq;
    std::list<DeferredPacket> pfqMissingTranslation;

    using const_iterator = std::list<DeferredPacket>::const_iterator;
    using iterator = std::list<DeferredPacket>::iterator;

    // PARAMETERS

    /** Maximum size of the prefetch queue */
    const unsigned queueSize;

    /**
     * Maximum size of the queue holding prefetch requests with missing
     * address translations
     */
    const unsigned missingTranslationQueueSize;

    /** Cycles after generation when a prefetch can first be issued */
    const Cycles latency;

    /** Squash queued prefetch if demand access observed */
    const bool queueSquash;

    /** Filter prefetches if already queued */
    const bool queueFilter;

    /** Snoop the cache before generating prefetch (cheating basically) */
    const bool cacheSnoop;

    /** Tag prefetch with PC of generating access? */
    const bool tagPrefetch;

    /** Percentage of requests that can be throttled */
    const unsigned int throttleControlPct;

    // STATS
    Stats::Scalar pfIdentified;
    Stats::Scalar pfBufferHit;
    Stats::Scalar pfInCache;
    Stats::Scalar pfRemovedFull;
    Stats::Scalar pfSpanPage;
    Stats::Scalar pfSquashed;

  public:
    using AddrPriority = std::pair<Addr, int32_t>;

    QueuedPrefetcher(const QueuedPrefetcherParams *p);
    virtual ~QueuedPrefetcher();

    void notify(const PacketPtr &pkt, const PrefetchInfo &pfi) override;

    void insert(const PacketPtr &pkt, PrefetchInfo &new_pfi, int32_t priority);

    virtual void calculatePrefetch(const PrefetchInfo &pfi,
                                   std::vector<AddrPriority> &addresses) = 0;
    PacketPtr getPacket() override;

    Tick nextPrefetchReadyTime() const override
    {
        return pfq.empty() ? MaxTick : pfq.front().tick;
    }

    void regStats() override;

  private:

    /**
     * Adds a DeferredPacket to the specified queue
     * @param queue selected queue to use
     * @param dpp DeferredPacket to add
     */
    void addToQueue(std::list<DeferredPacket> &queue, DeferredPacket &dpp);

    /**
     * Starts the translations of the queued prefetches with a
     * missing translation. It performs a maximum specified number of
     * translations. Successful translations cause the prefetch request to be
     * queued in the queue of ready requests.
     * @param max maximum number of translations to perform
     */
    void processMissingTranslations(unsigned max);

    /**
     * Indicates that the translation of the address of the provided  deferred
     * packet has been successfully completed, and it can be enqueued as a
     * new prefetch request.
     * @param dp the deferred packet that has completed the translation request
     * @param failed whether the translation was successful
     */
    void translationComplete(DeferredPacket *dp, bool failed);

    /**
     * Checks whether the specified prefetch request is already in the
     * specified queue. If the request is found, its priority is updated.
     * @param queue selected queue to check
     * @param pfi information of the prefetch request to be added
     * @param priority priority of the prefetch request to be added
     * @return True if the prefetch request was found in the queue
     */
    bool alreadyInQueue(std::list<DeferredPacket> &queue,
                        const PrefetchInfo &pfi, int32_t priority);

    /**
     * Returns the maxmimum number of prefetch requests that are allowed
     * to be created from the number of prefetch candidates provided.
     * The behavior of this service is controlled with the throttleControlPct
     * parameter.
     * @param total number of prefetch candidates generated by the prefetcher
     * @return the number of these request candidates are allowed to be created
     */
    size_t getMaxPermittedPrefetches(size_t total) const;

    RequestPtr createPrefetchRequest(Addr addr, PrefetchInfo const &pfi,
                                        PacketPtr pkt);
};

#endif //__MEM_CACHE_PREFETCH_QUEUED_HH__

