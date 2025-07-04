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

#include "mem/cache/replacement_policies/sp_rp.hh"
#include <cassert>
#include <memory>

#include "base/logging.hh" // For fatal_if
#include "base/random.hh"
#include "params/SPRP.hh"

SPRP::SPRP(const Params *p)
    : BRRIPRP(p)
{
    acc_str = p->accelerator_str;
    system = p->system;
    acc_id = system-> lookupMasterId(acc_str);
    promote_acc_wr = p->promote_acc_write;
    lines_promoted =0;
    wrong_promotions=0;
    promote_pf = p->promote_pf;
    mod_rrpv = p->mod_rrpv;
}
void
SPRP::invalidate(const std::shared_ptr<ReplacementData>& replacement_data)
const
{
    std::shared_ptr<SPReplData> casted_replacement_data =
        std::static_pointer_cast<SPReplData>(replacement_data);

    // Invalidate entry
    casted_replacement_data->valid = false;
}

void
SPRP::touch(const std::shared_ptr<ReplacementData>& replacement_data) const
{
    panic("SPRP reuires pkt info to function properly");
}

void
SPRP::touch(const std::shared_ptr<ReplacementData>& replacement_data, PacketPtr pkt) const
{
    std::shared_ptr<SPReplData> casted_replacement_data =
        std::static_pointer_cast<SPReplData>(replacement_data);

    
    if ( pkt->isWrite() && (pkt->masterId() == acc_id) && promote_acc_wr){
    casted_replacement_data->scan =true; 
    BRRIPRP::touch(replacement_data);
    //lines_promoted += lines_promoted;
    }
    else if ( casted_replacement_data -> scan == true){
        casted_replacement_data-> scan = false;
        casted_replacement_data->rrpv.saturate();
        if (mod_rrpv)
            casted_replacement_data->rrpv--;

    }
    else{
        BRRIPRP::touch(replacement_data);
    }
}

void
SPRP::reset(const std::shared_ptr<ReplacementData>& replacement_data) const
{
    panic("SPRP reuires pkt info to function properly");
}

void
SPRP::reset(const std::shared_ptr<ReplacementData>& replacement_data, const PacketPtr pkt) const
{
    std::shared_ptr<SPReplData> casted_replacement_data =
        std::static_pointer_cast<SPReplData>(replacement_data);

    // Reset RRPV
    // Replacement data is inserted as "long re-reference" if lower than btp,
    // "distant re-reference" otherwise
    if ( (pkt->req->isPrefetch() && promote_pf) || (pkt->isWrite() && (pkt->masterId() == acc_id) && promote_acc_wr)){
        casted_replacement_data->scan = true;
        //lines_promoted += lines_promoted;
        //incLines_promoted();
    }
    else{
        casted_replacement_data->scan = false;
    }
    BRRIPRP::reset(replacement_data);
    /*casted_replacement_data->rrpv.saturate();
    if (random_mt.random<unsigned>(1, 100) <= btp) {
        casted_replacement_data->rrpv--;
    }

    // Mark entry as ready to be used
    casted_replacement_data->valid = true;
    */
}

ReplaceableEntry*
SPRP::getVictim(const ReplacementCandidates& candidates) const
{
    // There must be at least one replacement candidate
    assert(candidates.size() > 0);

    // Use first candidate as dummy victim
    ReplaceableEntry* victim = nullptr; //= candidates[0];
    int pos =0;
    int victim_RRPV; 
    for (const auto& candidate: candidates){
        std::shared_ptr<SPReplData> candidate_repl_data =
        std::static_pointer_cast<SPReplData>(
            candidate->replacementData);
            if ( candidate_repl_data->valid == false){
                return candidate;

            }
            else if ( candidate_repl_data-> scan == false){
                victim = candidate;
                break;
            }
            else 
                pos++;
    }
    if ( victim == nullptr){
        for (const auto& candidate: candidates){
            std::shared_ptr<SPReplData> candidate_repl_data =
            std::static_pointer_cast<SPReplData>(
                candidate->replacementData);
                candidate_repl_data->scan = false;
        }
        victim = candidates[0];
        pos =0;
        //wrong_promotions += wrong_promotions;
    }
    // Store victim->rrpv in a variable to improve code readability
    victim_RRPV= std::static_pointer_cast<SPReplData>(
                        victim->replacementData)->rrpv;

    // Visit all candidates to find victim
    for (auto itr = (candidates.begin() + pos ); itr != candidates.end(); itr++) {
        auto candidate = *itr;
        std::shared_ptr<SPReplData> candidate_repl_data =
            std::static_pointer_cast<SPReplData>(
                candidate->replacementData);

        // Stop searching for victims if an invalid entry is found
        /*if (!candidate_repl_data->valid) {
            return candidate;
        }*/

        // Update victim entry if necessary
        int candidate_RRPV = candidate_repl_data->rrpv;
        if ((candidate_RRPV > victim_RRPV ) && candidate_repl_data->scan ==false ) {
            victim = candidate;
            victim_RRPV = candidate_RRPV;
        }
    }

    // Get difference of victim's RRPV to the highest possible RRPV in
    // order to update the RRPV of all the other entries accordingly
    int diff = std::static_pointer_cast<SPReplData>(
        victim->replacementData)->rrpv.saturate();

    // No need to update RRPV if there is no difference
    if (diff > 0){
        // Update RRPV of all candidates
        for (const auto& candidate : candidates) {
            std::static_pointer_cast<SPReplData>(
                candidate->replacementData)->rrpv += diff;
        }
    }

    return victim;
}

std::shared_ptr<ReplacementData>
SPRP::instantiateEntry()
{
    return std::shared_ptr<ReplacementData>(new SPReplData(numRRPVBits));
}

void SPRP::regStats(){

    lines_promoted
        .name(name() + ".lines_promoted")
        .desc("number of scan lines promoted to high priority");

    wrong_promotions
        .name(name() + ".wrong_promotions")
        .desc("number of times promoted lines filled all the ways leading to demoting all ways");
}

SPRP*
SPRPParams::create()
{
    return new SPRP(this);
}
