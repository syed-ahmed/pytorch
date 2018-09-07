#include "Generator.h"

#include <structmember.h>
#include <ATen/ATen.h>

#include <stdbool.h>
#include <TH/TH.h>
#include "THP.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/autograd/generated/VariableType.h"
#include "torch/csrc/utils/tensor_types.h"
#include "torch/csrc/utils/python_arg_parser.h"
#include "torch/csrc/autograd/generated/variable_factories.h"

using namespace at;
using namespace torch;

PyObject *THPGeneratorClass = nullptr;

PyObject * THPGenerator_New()
{
  PyObject *args = PyTuple_New(0);
  if (!args) {
    PyErr_SetString(PyExc_RuntimeError, "Could not create a new generator object - "
        "failed to allocate argument tuple");
    return nullptr;
  }
  PyObject *result = PyObject_Call((PyObject*)THPGeneratorClass, args, nullptr);
  Py_DECREF(args);
  return result;
}

PyObject * THPGenerator_NewWithGenerator(at::Generator& cdata)
{
  auto type = (PyTypeObject*)THPGeneratorClass;
  auto self = THPObjectPtr{type->tp_alloc(type, 0)};
  if (!self) throw python_error();
  auto self_ = reinterpret_cast<THPGenerator*>(self.get());
  self_->cdata = &cdata;
  return self.release();
}

static void THPGenerator_dealloc(THPGenerator* self)
{
  if (self->owner) {
    delete self->cdata;
  }
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject * THPGenerator_pynew(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser({
    "Generator(Device device)"
  });
  torch::ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto as_device = r.deviceWithDefault(0, at::Device(at::kCPU));
  auto device_type = r.string(0);
  if (as_device.has_index()) {
    throw std::runtime_error("type (string) must not include an index because index "
                              "was passed explicitly: " + device_type);
  }
  int32_t device_index = -1;
  if (!r.isNone(1)) {
    device_index = r.toInt64(1);
    // -1 is allowed in ATen/C++, to mean the default device, but not in
    // Python.
    AT_CHECK(device_index >= 0, "Device index must not be negative");
  }
  THPGeneratorPtr self((THPGenerator *)type->tp_alloc(type, 0));
  self->cdata = at::globalContext().createGenerator(as_device.type(), device_index);
  self->owner = true;
  return (PyObject*)self.release();
  END_HANDLE_TH_ERRORS
}

static PyObject * THPGenerator_getState(PyObject *self, PyObject *args, PyObject *kwargs)
{
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser({
    "get_state(int64_t? device=-1)"
  });
  torch::ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  int32_t device_index = -1;
  if (!r.isNone(0)) {
    device_index = r.toInt64(0);
    // -1 is allowed in ATen/C++, to mean the default device, but not in
    // Python.
    AT_CHECK(device_index >= 0, "Device index must not be negative");
  }
  Variable var = self->cdata()->getState(device_index);
  return THPVariable_Wrap(std::move(var));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPGenerator_setState(THPGenerator *self, PyObject *_new_state, PyObject *args, PyObject *kwargs)
{
  HANDLE_TH_ERRORS
  if (!THPVariable_Check(_new_state)) {
    throw TypeError("expected a GeneratorState, but got %s", Py_TYPE(_new_state)->tp_name);
  }
  static torch::PythonArgParser parser({
    "set_state(*, int64_t? device=-1)"
  });
  torch::ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  int32_t device_index = -1;
  if (!r.isNone(0)) {
    device_index = r.toInt64(0);
    // -1 is allowed in ATen/C++, to mean the default device, but not in
    // Python.
    AT_CHECK(device_index >= 0, "Device index must not be negative");
  }
  auto& gen_state = ((THPVariable*)_new_state)->cdata.data();
  self->cdata()->setState(gen_state, device_index);
  Py_INCREF(self);
  return (PyObject*)self;
  END_HANDLE_TH_ERRORS
}

static PyObject * THPGenerator_manualSeed(THPGenerator *self, PyObject *seed)
{
  HANDLE_TH_ERRORS
  auto generator = self->cdata;
  THPUtils_assert(THPUtils_checkLong(seed), "manual_seed expected a long, "
          "but got %s", THPUtils_typename(seed));
  generator->manualSeed(THPUtils_unpackLong(seed));
  Py_INCREF(self);
  return (PyObject*)self;
  END_HANDLE_TH_ERRORS
}

static PyObject * THPGenerator_seed(THPGenerator *self)
{
  HANDLE_TH_ERRORS
  return THPUtils_packUInt64(self->cdata->seed());
  END_HANDLE_TH_ERRORS
}

static PyObject * THPGenerator_initialSeed(THPGenerator *self)
{
  HANDLE_TH_ERRORS
  return THPUtils_packUInt64(self->cdata->getStartingSeed());
  END_HANDLE_TH_ERRORS
}

static PyMethodDef THPGenerator_methods[] = {
  {"get_state",       (PyCFunction)THPGenerator_getState,       METH_O,  nullptr},
  {"set_state",       (PyCFunction)THPGenerator_setState,       METH_O,       nullptr},
  {"manual_seed",     (PyCFunction)THPGenerator_manualSeed,     METH_O,       nullptr},
  {"manual_seed_all", (PyCFunction)THCPModule_manualSeedAll,    METH_O,       nullptr},
  {"seed",            (PyCFunction)THPGenerator_seed,           METH_NOARGS,  nullptr},
  {"seed_all",        (PyCFunction)THCPModule_seedAll,          METH_NOARGS,  nullptr},
  {"initial_seed",    (PyCFunction)THPGenerator_initialSeed,    METH_NOARGS,  nullptr},
  {nullptr}
};

static struct PyMemberDef THPGenerator_members[] = {
  {(char*)"_cdata", T_ULONGLONG, offsetof(THPGenerator, cdata), READONLY, nullptr},
  {nullptr}
};

PyTypeObject THPGeneratorType = {
  PyVarObject_HEAD_INIT(nullptr, 0)
  "torch._C.Generator",                  /* tp_name */
  sizeof(THPGenerator),                  /* tp_basicsize */
  0,                                     /* tp_itemsize */
  (destructor)THPGenerator_dealloc,      /* tp_dealloc */
  0,                                     /* tp_print */
  0,                                     /* tp_getattr */
  0,                                     /* tp_setattr */
  0,                                     /* tp_reserved */
  0,                                     /* tp_repr */
  0,                                     /* tp_as_number */
  0,                                     /* tp_as_sequence */
  0,                                     /* tp_as_mapping */
  0,                                     /* tp_hash  */
  0,                                     /* tp_call */
  0,                                     /* tp_str */
  0,                                     /* tp_getattro */
  0,                                     /* tp_setattro */
  0,                                     /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
  nullptr,                                  /* tp_doc */
  0,                                     /* tp_traverse */
  0,                                     /* tp_clear */
  0,                                     /* tp_richcompare */
  0,                                     /* tp_weaklistoffset */
  0,                                     /* tp_iter */
  0,                                     /* tp_iternext */
  THPGenerator_methods,                  /* tp_methods */
  THPGenerator_members,                  /* tp_members */
  0,                                     /* tp_getset */
  0,                                     /* tp_base */
  0,                                     /* tp_dict */
  0,                                     /* tp_descr_get */
  0,                                     /* tp_descr_set */
  0,                                     /* tp_dictoffset */
  0,                                     /* tp_init */
  0,                                     /* tp_alloc */
  THPGenerator_pynew,                    /* tp_new */
};

bool THPGenerator_init(PyObject *module)
{
  THPGeneratorClass = (PyObject*)&THPGeneratorType;
  if (PyType_Ready(&THPGeneratorType) < 0)
    return false;
  Py_INCREF(&THPGeneratorType);
  PyModule_AddObject(module, "Generator", (PyObject *)&THPGeneratorType);
  return true;
}
