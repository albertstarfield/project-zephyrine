#ifndef INTERPRETER_MANAGER_H
#define INTERPRETER_MANAGER_H

#include <pybind11/embed.h>
#include <memory>

namespace py = pybind11;

class InterpreterManager {
public:
    static void init();

private:
    static std::unique_ptr<py::scoped_interpreter> guard;
};

#endif // INTERPRETER_MANAGER_H
