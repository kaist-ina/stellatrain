#include "barrier_checker.h"
#include "../core.h"

#if DEBUG_BARRIER
ModuleRunResult ModuleBarrierChecker::run(FasterDpEngine *engine, TrainTaskV2 *task) {
    if (task->test_field_ >= 60) {
        std::cerr << "Found no problem, terminating..." << std::endl;
        engine->schedule_after_use(engine->module_d2h_copy_, task);
    } else {
        std::stringstream s;
        s << "[Task " << task->key() << "] : ";
        int idx_to_check = engine->local_rank() == 0 ? 1 : 0;
        if (task->test_field_ % 2 == 0) {
            // set
            // s << " setting task->test_field_ = " << task->test_field_;
            task->shared_props_->shared_test_field_[engine->local_rank()] = task->test_field_;
        } else {
            // verify
            // s << " verifying remote task->test_field_ = " << (task->test_field_ - 1);
            int target = task->shared_props_->shared_test_field_[idx_to_check];
            assert(target == task->test_field_ - 1);
        }
        task->test_field_ += 1;
        // std::cerr << s.str() << std::endl;

        engine->schedule_after_barrier(engine->module_barrier_checker_, task, module_name() + std::to_string(task->test_field_));
    }

    return ModuleRunResult::RESULT_SUCCESS;
}

#endif