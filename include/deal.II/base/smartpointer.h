// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 1998 - 2023 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------

#ifndef dealii_smartpointer_h
#define dealii_smartpointer_h


#include <deal.II/base/config.h>

#include <deal.II/base/exceptions.h>

#include <atomic>
#include <typeinfo>

DEAL_II_NAMESPACE_OPEN

/**
 * The SmartPointer class avoids using dangling pointers. They can be used just
 * like a pointer (i.e., using the <tt>*</tt> and <tt>-></tt> operators and
 * through casting) but make sure that the object pointed to is not deleted or
 * moved from in the course of use of the pointer by signaling the pointee its
 * use.
 *
 * Conceptually, SmartPointer fills a gap between `std::unique_ptr` and
 * `std::shared_ptr`. While the former makes it clear that there is a unique
 * owner of an object (namely the scope in which the `std::unique_ptr` resides),
 * it does not allow other places in a code base to point to the object. In
 * contrast, `std::shared_ptr` allows many places to point to the same object,
 * but none of them is the "owner" of the object: They all are, and the last
 * one to stop pointing to the object is responsible for deleting it.
 *
 * SmartPointer utilizes semantics in which one place owns an object and others
 * can point to it. The owning place is responsible for destroying the object
 * when it is no longer needed, and this will trigger an error if other places
 * are still pointing to it via SmartPointer pointers. In other words, one
 * should consider those places that hold a SmartPointer to an object as
 * "observers", and an object may only be destroyed without an error if there
 * are no observers left. With hindsight, perhaps a better name for the class
 * would have been "ObserverPointer".
 *
 * To make this scheme work, SmartPointers need to increment the "observer
 * count" when they point to an observed object. This is facilitated by
 * requiring that the objects pointed to, i.e., the template type `T`, must
 * inherit from the Subscriptor class.
 *
 * In practice, using this scheme, if you try to destroy an object to which
 * observers still point via SmartPointer objects, you will get an error that
 * says that there are still observers of the object and that the object can
 * consequently not be destroyed without creating "dangling" pointers. This
 * is often not very helpful in finding where these pointers are. As a
 * consequence, this class also provides two ways to annotate the observer
 * count with information about what other places still observe an object.
 * First, when initializing a SmartPointer object with the address of the
 * object pointed to, you can also attach a string that describes the
 * observing location, and this string will then be shown in error messages
 * listing all remaining observers. Second, if no such string is provided,
 * the name second template argument `P` is used as the debug string. This
 * allows to encode the observer information in the type of the SmartPointer.
 *
 * @note Unlike `std::unique_ptr` and `std::shared_ptr`, SmartPointer does
 *   NOT implement any memory handling. In particular, deleting a
 *   SmartPointer does not delete the object because the semantics
 *   of this class are that it only *observes* an object, but does not
 *   take over ownership. As a consequence, this is a sure way of creating
 *   a memory leak:
 *   @code
 *     SmartPointer<T> dont_do_this = new T;
 *   @endcode
 *   This is because here, no variable "owns" the object pointed to, and
 *   the destruction of the `dont_do_this` pointer does not trigger the
 *   release of the memory pointed to.
 *
 * @note This class correctly handles `const`-ness of an object, i.e.,
 *   a `SmartPointer<const T>` really behaves as if it were a pointer
 *   to a constant object (disallowing write access when dereferenced), while
 *   `SmartPointer<T>` is a mutable pointer.
 *
 * @ingroup memory
 */
template <typename T, typename P = void>
class SmartPointer
{
public:
  /**
   * Standard constructor for null pointer. The id of this pointer is set to
   * the name of the class P.
   */
  SmartPointer();

  /**
   * Copy constructor for SmartPointer. We do not copy the object subscribed
   * to from <tt>tt</tt>, but subscribe ourselves to it again.
   */
  template <class Q>
  SmartPointer(const SmartPointer<T, Q> &tt);

  /**
   * Copy constructor for SmartPointer. We do not copy the object subscribed
   * to from <tt>tt</tt>, but subscribe ourselves to it again.
   */
  SmartPointer(const SmartPointer<T, P> &tt);

  /**
   * Constructor taking a normal pointer. If possible, i.e. if the pointer is
   * not a null pointer, the constructor subscribes to the given object to
   * lock it, i.e. to prevent its destruction before the end of its use.
   *
   * The <tt>id</tt> is used in the call to Subscriptor::subscribe(id) and by
   * ~SmartPointer() in the call to Subscriptor::unsubscribe().
   */
  SmartPointer(T *t, const std::string &id);

  /**
   * Constructor taking a normal pointer. If possible, i.e. if the pointer is
   * not a null pointer, the constructor subscribes to the given object to
   * lock it, i.e. to prevent its destruction before the end of its use. The
   * id of this pointer is set to the name of the class P.
   */
  SmartPointer(T *t);

  /**
   * Destructor, removing the subscription.
   */
  ~SmartPointer();

  /**
   * Assignment operator for normal pointers. The pointer subscribes to the
   * new object automatically and unsubscribes to an old one if it exists. It
   * will not try to subscribe to a null-pointer, but still delete the old
   * subscription.
   */
  SmartPointer<T, P> &
  operator=(T *tt);

  /**
   * Assignment operator for SmartPointer. The pointer subscribes to the new
   * object automatically and unsubscribes to an old one if it exists.
   */
  template <class Q>
  SmartPointer<T, P> &
  operator=(const SmartPointer<T, Q> &tt);

  /**
   * Assignment operator for SmartPointer. The pointer subscribes to the new
   * object automatically and unsubscribes to an old one if it exists.
   */
  SmartPointer<T, P> &
  operator=(const SmartPointer<T, P> &tt);

  /**
   * Delete the object pointed to and set the pointer to zero.
   */
  void
  clear();

  /**
   * Conversion to normal pointer.
   */
  operator T *() const;

  /**
   * Dereferencing operator. This operator throws an ExcNotInitialized() if the
   * pointer is a null pointer.
   */
  T &
  operator*() const;

  /**
   * Return underlying pointer. This operator throws an ExcNotInitialized() if
   * the pointer is a null pointer.
   */
  T *
  get() const;

  /**
   * Operator that returns the underlying pointer. This operator throws an
   * ExcNotInitialized() if the pointer is a null pointer.
   */
  T *
  operator->() const;

  /**
   * Exchange the pointers of this object and the argument. Since both the
   * objects to which is pointed are subscribed to before and after, we do not
   * have to change their subscription counters.
   *
   * Note that this function (with two arguments) and the respective functions
   * where one of the arguments is a pointer and the other one is a C-style
   * pointer are implemented in global namespace.
   */
  template <class Q>
  void
  swap(SmartPointer<T, Q> &tt);

  /**
   * Swap pointers between this object and the pointer given. As this releases
   * the object pointed to presently, we reduce its subscription count by one,
   * and increase it at the object which we will point to in the future.
   *
   * Note that we indeed need a reference of a pointer, as we want to change
   * the pointer variable which we are given.
   */
  void
  swap(T *&tt);

  /**
   * Return an estimate of the amount of memory (in bytes) used by this class.
   * Note in particular, that this only includes the amount of memory used by
   * <b>this</b> object, not by the object pointed to.
   */
  std::size_t
  memory_consumption() const;

private:
  /**
   * Pointer to the object we want to subscribe to. Since it is often
   * necessary to follow this pointer when debugging, we have deliberately
   * chosen a short name.
   */
  T *t;

  /**
   * The identification for the subscriptor.
   */
  const std::string id;

  /**
   * The Smartpointer is invalidated when the object pointed to is destroyed
   * or moved from.
   */
  std::atomic<bool> pointed_to_object_is_alive;
};


/* --------------------- inline Template functions ------------------------- */


template <typename T, typename P>
inline SmartPointer<T, P>::SmartPointer()
  : t(nullptr)
  , id(typeid(P).name())
  , pointed_to_object_is_alive(false)
{}



template <typename T, typename P>
inline SmartPointer<T, P>::SmartPointer(T *t)
  : t(t)
  , id(typeid(P).name())
  , pointed_to_object_is_alive(false)
{
  if (t != nullptr)
    t->subscribe(&pointed_to_object_is_alive, id);
}



template <typename T, typename P>
inline SmartPointer<T, P>::SmartPointer(T *t, const std::string &id)
  : t(t)
  , id(id)
  , pointed_to_object_is_alive(false)
{
  if (t != nullptr)
    t->subscribe(&pointed_to_object_is_alive, id);
}



template <typename T, typename P>
template <class Q>
inline SmartPointer<T, P>::SmartPointer(const SmartPointer<T, Q> &tt)
  : t(tt.t)
  , id(tt.id)
  , pointed_to_object_is_alive(false)
{
  if (tt.pointed_to_object_is_alive && t != nullptr)
    t->subscribe(&pointed_to_object_is_alive, id);
}



template <typename T, typename P>
inline SmartPointer<T, P>::SmartPointer(const SmartPointer<T, P> &tt)
  : t(tt.t)
  , id(tt.id)
  , pointed_to_object_is_alive(false)
{
  if (tt.pointed_to_object_is_alive && t != nullptr)
    t->subscribe(&pointed_to_object_is_alive, id);
}



template <typename T, typename P>
inline SmartPointer<T, P>::~SmartPointer()
{
  if (pointed_to_object_is_alive && t != nullptr)
    t->unsubscribe(&pointed_to_object_is_alive, id);
}



template <typename T, typename P>
inline void
SmartPointer<T, P>::clear()
{
  if (pointed_to_object_is_alive && t != nullptr)
    {
      t->unsubscribe(&pointed_to_object_is_alive, id);
      delete t;
      Assert(pointed_to_object_is_alive == false, ExcInternalError());
    }
  t = nullptr;
}



template <typename T, typename P>
inline SmartPointer<T, P> &
SmartPointer<T, P>::operator=(T *tt)
{
  // optimize if no real action is
  // requested
  if (t == tt)
    return *this;

  if (pointed_to_object_is_alive && t != nullptr)
    t->unsubscribe(&pointed_to_object_is_alive, id);
  t = tt;
  if (tt != nullptr)
    tt->subscribe(&pointed_to_object_is_alive, id);
  return *this;
}



template <typename T, typename P>
template <class Q>
inline SmartPointer<T, P> &
SmartPointer<T, P>::operator=(const SmartPointer<T, Q> &tt)
{
  // if objects on the left and right
  // hand side of the operator= are
  // the same, then this is a no-op
  if (&tt == this)
    return *this;

  if (pointed_to_object_is_alive && t != nullptr)
    t->unsubscribe(&pointed_to_object_is_alive, id);
  t = static_cast<T *>(tt);
  if (tt.pointed_to_object_is_alive && tt != nullptr)
    tt->subscribe(&pointed_to_object_is_alive, id);
  return *this;
}



template <typename T, typename P>
inline SmartPointer<T, P> &
SmartPointer<T, P>::operator=(const SmartPointer<T, P> &tt)
{
  // if objects on the left and right
  // hand side of the operator= are
  // the same, then this is a no-op
  if (&tt == this)
    return *this;

  if (pointed_to_object_is_alive && t != nullptr)
    t->unsubscribe(&pointed_to_object_is_alive, id);
  t = static_cast<T *>(tt);
  if (tt.pointed_to_object_is_alive && tt != nullptr)
    tt->subscribe(&pointed_to_object_is_alive, id);
  return *this;
}



template <typename T, typename P>
inline SmartPointer<T, P>::operator T *() const
{
  return t;
}



template <typename T, typename P>
inline T &
SmartPointer<T, P>::operator*() const
{
  Assert(t != nullptr, ExcNotInitialized());
  Assert(pointed_to_object_is_alive,
         ExcMessage("The object pointed to is not valid anymore."));
  return *t;
}



template <typename T, typename P>
inline T *
SmartPointer<T, P>::get() const
{
  Assert(t != nullptr, ExcNotInitialized());
  Assert(pointed_to_object_is_alive,
         ExcMessage("The object pointed to is not valid anymore."));
  return t;
}



template <typename T, typename P>
inline T *
SmartPointer<T, P>::operator->() const
{
  return this->get();
}



template <typename T, typename P>
template <class Q>
inline void
SmartPointer<T, P>::swap(SmartPointer<T, Q> &tt)
{
#ifdef DEBUG
  SmartPointer<T, P> aux(t, id);
  *this = tt;
  tt    = aux;
#else
  std::swap(t, tt.t);
#endif
}



template <typename T, typename P>
inline void
SmartPointer<T, P>::swap(T *&tt)
{
  if (pointed_to_object_is_alive && t != nullptr)
    t->unsubscribe(pointed_to_object_is_alive, id);

  std::swap(t, tt);

  if (t != nullptr)
    t->subscribe(pointed_to_object_is_alive, id);
}



template <typename T, typename P>
inline std::size_t
SmartPointer<T, P>::memory_consumption() const
{
  return sizeof(SmartPointer<T, P>);
}



// The following function is not strictly necessary but is an optimization
// for places where you call swap(p1,p2) with SmartPointer objects p1, p2.
// Unfortunately, MS Visual Studio (at least up to the 2013 edition) trips
// over it when calling std::swap(v1,v2) where v1,v2 are std::vectors of
// SmartPointer objects: it can't determine whether it should call std::swap
// or dealii::swap on the individual elements (see bug #184 on our Google Code
// site. Consequently, just take this function out of the competition for this
// compiler.
#ifndef _MSC_VER
/**
 * Global function to swap the contents of two smart pointers. As both objects
 * to which the pointers point retain to be subscribed to, we do not have to
 * change their subscription count.
 */
template <typename T, typename P, class Q>
inline void
swap(SmartPointer<T, P> &t1, SmartPointer<T, Q> &t2)
{
  t1.swap(t2);
}
#endif


/**
 * Global function to swap the contents of a smart pointer and a C-style
 * pointer.
 *
 * Note that we indeed need a reference of a pointer, as we want to change the
 * pointer variable which we are given.
 */
template <typename T, typename P>
inline void
swap(SmartPointer<T, P> &t1, T *&t2)
{
  t1.swap(t2);
}



/**
 * Global function to swap the contents of a C-style pointer and a smart
 * pointer.
 *
 * Note that we indeed need a reference of a pointer, as we want to change the
 * pointer variable which we are given.
 */
template <typename T, typename P>
inline void
swap(T *&t1, SmartPointer<T, P> &t2)
{
  t2.swap(t1);
}

DEAL_II_NAMESPACE_CLOSE

#endif
