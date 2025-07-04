/**
 * Copyright (c) 2018 Inria
 * All rights reserved.
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
 * Authors: Daniel Carvalho
 */

/**
 * @file
 * Declaration of a Re-Reference Interval Prediction replacement policy.
 *
 * Not-Recently Used (NRU) is an approximation of LRU that uses a single bit
 * to determine if an entry is going to be re-referenced in the near or distant
 * future.
 *
 * Re-Reference Interval Prediction (RRIP) is an extension of NRU that uses a
 * re-reference prediction value to determine if entries are going to be re-
 * used in the near future or not.
 *
 * The higher the value of the RRPV, the more distant the entry is from its
 * next access.
 *
 * Bimodal Re-Reference Interval Prediction (BRRIP) is an extension of RRIP
 * that has a probability of not inserting entries as the LRU. This probability
 * is controlled by the bimodal throtle parameter (btp).
 *
 * From the original paper, this implementation of RRIP is also called
 * Static RRIP (SRRIP), as it always inserts entries with the same RRPV.
 */

#ifndef __MEM_CACHE_REPLACEMENT_POLICIES_SP_RP_HH__
#define __MEM_CACHE_REPLACEMENT_POLICIES_SP_RP_HH__

#include "base/sat_counter.hh"
#include "mem/cache/replacement_policies/base.hh"
#include "mem/cache/replacement_policies/brrip_rp.hh"
#include <unordered_set>
#include "mem/packet.hh"
#include "mem/request.hh"
#include "sim/system.hh"
#include "base/statistics.hh"
#include "base/types.hh"
//Scan promotion policy

struct SPRPParams;

class SPRP : public BRRIPRP
{
  protected:
    /** BRRIP-specific implementation of replacement data. */
    struct SPReplData : BRRIPReplData
    {
        
        bool scan;
        /**
         * Default constructor. Invalidate data.
         */
        SPReplData(const int num_bits)
            : BRRIPReplData(num_bits), scan(false)
        {
        }
    };

    //std::unordered_set<int> master_ids;
    Stats::Scalar lines_promoted ;
    Stats::Scalar wrong_promotions ;

    void incLines_promoted(){
      lines_promoted++;
    }
  public:
    /** Convenience typedef. */
    typedef SPRPParams Params;

    /**
     * Construct and initiliaze this replacement policy.
     */
    SPRP(const Params *p);

    /**
     * Destructor.
     */
    ~SPRP() {}

    /**
     * Invalidate replacement data to set it as the next probable victim.
     * Set RRPV as the the most distant re-reference.
     *
     * @param replacement_data Replacement data to be invalidated.
     */
    void invalidate(const std::shared_ptr<ReplacementData>& replacement_data)
                                                              const override;

    /**
     * Touch an entry to update its replacement data.
     *
     * @param replacement_data Replacement data to be touched.
     */
    void touch(const std::shared_ptr<ReplacementData>& replacement_data) const
                                                                     override;
    void touch(const std::shared_ptr<ReplacementData>& replacement_data, const PacketPtr pkt) const
                                                                     override;
    /**
     * Reset replacement data. Used when an entry is inserted.
     * Set RRPV according to the insertion policy used.
     *
     * @param replacement_data Replacement data to be reset.
     */
    void reset(const std::shared_ptr<ReplacementData>& replacement_data) const
                                                                     override;
    void reset(const std::shared_ptr<ReplacementData>&
                                                replacement_data, const PacketPtr pkt) const override;
    /**
     * Find replacement victim using rrpv.
     *
     * @param cands Replacement candidates, selected by indexing policy.
     * @return Replacement entry to be replaced.
     */
    ReplaceableEntry* getVictim(const ReplacementCandidates& candidates) const
                                                                     override;

    /**
     * Instantiate a replacement data entry.
     *
     * @return A shared pointer to the new replacement data.
     */
    std::shared_ptr<ReplacementData> instantiateEntry() override;

    void regStats() override;

    std::string acc_str ;
    MasterID acc_id;
    System* system;
    bool promote_acc_wr;
    bool promote_pf;
    bool mod_rrpv;
    
};

#endif // __MEM_CACHE_REPLACEMENT_POLICIES_BRRIP_RP_HH__
