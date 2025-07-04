#include <iostream>
#include <string>
#include <vector>

#include "smaug/utility/debug_stream.h"
#include "smaug/utility/thread_pool.h"
#include "smaug/core/tensor.h"
#include "smaug/core/types.pb.h"
#include "smaug/core/scheduler.h"

namespace smaug {

Tensor* Scheduler::runNetwork(int iterations) {
    std::cout << "======================================================\n";
    std::cout << "      Tiling operators of the network...\n";
    std::cout << "======================================================\n";
    for (auto nameOp : network->getOperators()) {
        Operator* op = nameOp.second;
        dout(0) << "Tiling " << op->getName() << " ("
                << OpType_Name(op->getOpType()) << ").\n";
        op->tile();
    }
    dout(0) << " total num of tiles " << stat_num_of_tiles << " \n";
    // We have finished loading the model and building the network, as well as
    // the tiling of all the operators. Now we can stop fast forwarding.
    gem5::switchCpu();

    fastForwardMode = false;

    // The fast-forwarding mode uses simpler CPUs, which will be switched to
    // OoO CPUs after it's done. Therefore, the initialization of the thread
    // pool must be after the fast-forwarding, otherwise the CPU IDs will be
    // incorrect.
    if (threadPool)
        threadPool->initThreadPool();
    Tensor* output;
    gem5::resetStats(0); 
    for ( int loop = 0; loop < iterations ; loop ++){
        std::cout << "======================================================\n";
        std::cout << "      Scheduling operators of the network...\n";
        std::cout << "      Iteration : " << loop << " \n";
        std::cout << "======================================================\n";
        // Initialize number of pending inputs for every operator and put Data
        // operators into the ready queue.
        {
            //auto stats =
              //     gem5::ScopedStats(stats::kNetworkStart, stats::kNetworkEnd)
        for (auto nameOp : network->getOperators()) {
            Operator* op = nameOp.second;
            Vertex vertex = op->getVertex();
            int numPendingInputs = boost::in_degree(vertex, network->getGraph());
            if ( loop == 0)
                op->setNumPendingInputs(numPendingInputs);
            else
                op->setNumPendingInputs(1); // after the first time, the data_op inputs would already be available.         
            if (numPendingInputs == 0)
                readyQueue.push_back(op);
        }
        
        
            /*//auto stats =
            //        gem5::ScopedStats(stats::kNetworkStart, stats::kNetworkEnd);*/
            
                output = scheduleReady();
        }
        readyQueue.clear(); 
        std::string stat_string = " stats after iteration " + std::to_string(loop);
        //gem5::dumpResetStats( (const char *)stat_string.c_str(), 0);
    }
    gem5::dumpResetStats( "Iteration Ends", 0);

    return output;
}

Tensor* Scheduler::scheduleReady() {
    Tensor* output;
    
    for (auto op : readyQueue) {
        dout(0) << "Scheduling " << op->getName() << " ("
                << OpType_Name(op->getOpType()) << ").\n";
        maybeRunOperator(op);
        updateChildren(op);
        output = op->getOutput(0);
        dout(2) << *output << "\n";
    }
    return output;
}

void Scheduler::maybeRunOperator(Operator* op) {
    uint64_t curr_tick =0; 
    double latency =0;
    static double cumm_latency =0;
    static uint64_t num_operators =0;
    double latency_per_operator = 0.0;
    if (!op->isDead()) {
        dout(1) << op->getName() << " running \n";
        if (op->getOpType() != OpType::Data ){
            curr_tick = gem5::get_curr_Tick();
            op->run();
            latency = double(gem5::get_curr_Tick() - curr_tick);
            cumm_latency += latency;
            num_operators ++;
            latency_per_operator = cumm_latency/num_operators; 
            dout (0) << op->getName() << " latency : " << latency << " ns \n";
            dout (0) << " Operators executed : " << num_operators << "   Latency per operator : " << latency_per_operator << " \n"; 
            //if ( op->getName().find("chkpt") != -1 ){
                gem5::dumpStats(op->getName().c_str(), 0);
            //}
        }
        else {
            op->run();
        }

    } else {
        dout(1) << op->getName() << " is dead \n";
        for (auto output : op->getOutputs())
            output->setDead();
    }
}

void Scheduler::updateChildren(Operator* op) {
    const Graph& graph = network->getGraph();
    Vertex vertex = op->getVertex();
    out_edge_iter outEdgeIt, outEdgeEnd;
    for (boost::tie(outEdgeIt, outEdgeEnd) = out_edges(vertex, graph);
         outEdgeIt != outEdgeEnd;
         ++outEdgeIt) {
        Vertex childVertex = target(*outEdgeIt, graph);
        Operator* child = get(boost::vertex_op, graph, childVertex);
        if (child->getNumPendingInputs() > 0) {
            child->decrNumPendingInputs();
            if (child->getNumPendingInputs() == 0)
                readyQueue.push_back(child);
        }
    }
}

}  // namespace smaug
