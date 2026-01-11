/*
  Copyright (c) 2009-2017 Dave Gamble and cJSON contributors

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  THE SOFTWARE.
*/

/* cJSON */
/* JSON parser in C. */

#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <limits.h>
#include <ctype.h>
#include <float.h>

#ifdef ENABLE_LOCALES
#include <locale.h>
#endif

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-qual"
#endif

#include "cJSON.h"

// Simple strcasecmp implementation for platforms where it might be missing or not in string.h
static int cJSON_strcasecmp(const unsigned char *s1, const unsigned char *s2) {
    if (!s1 || !s2) {
        return (s1 == s2) ? 0 : ((s1 > s2) ? 1 : -1);
    }
    for (; *s1 && tolower(*s1) == tolower(*s2); ++s1, ++s2);
    return tolower(*s1) - tolower(*s2);
}

/* define our own boolean type */
#define internal_IsTrue  1
#define internal_IsFalse 0

typedef struct internal_hooks
{
    void *(*allocate)(size_t size);
    void (*deallocate)(void *pointer);
    void *(*reallocate)(void *pointer, size_t size);
} internal_hooks;

#if defined(_MSC_VER)
/* Visual Studio */
#pragma warning (push)
#pragma warning (disable : 4996)
#endif

static internal_hooks global_hooks;

static unsigned char *cJSON_strdup(const unsigned char *string)
{
    size_t length = 0;
    unsigned char *copy = NULL;

    if (string == NULL)
    {
        return NULL;
    }

    length = strlen((const char*)string) + sizeof("");
    copy = (unsigned char*)global_hooks.allocate(length);
    if (copy == NULL)
    {
        return NULL;
    }
    memcpy(copy, string, length);

    return copy;
}

void cJSON_InitHooks(cJSON_Hooks* hooks)
{
    if (hooks == NULL)
    {
        /* Reset to using default malloc/free */
        global_hooks.allocate = malloc;
        global_hooks.deallocate = free;
        global_hooks.reallocate = realloc;
        return;
    }

    global_hooks.allocate = hooks->malloc_fn;
    global_hooks.deallocate = hooks->free_fn;
    global_hooks.reallocate = NULL;
}

/* Reset the cJSON_Hooks structure to the default malloc/free functions. */
void cJSON_ResetHooks(void)
{
    cJSON_InitHooks(NULL);
}

/* Internal constructor. */
static cJSON *cJSON_New_Item(const internal_hooks * const hooks)
{
    cJSON* node = (cJSON*)hooks->allocate(sizeof(cJSON));
    if (node)
    {
        memset(node, '\0', sizeof(cJSON));
    }

    return node;
}

/* Delete a cJSON entity and all its children. */
void cJSON_Delete(cJSON *item)
{
    cJSON *next = NULL;
    while (item)
    {
        next = item->next;
        if (item->child)
        {
            cJSON_Delete(item->child);
        }
        if (!(item->type & cJSON_IsReference) && (item->valuestring != NULL))
        {
            global_hooks.deallocate(item->valuestring);
        }
        if (!(item->type & cJSON_StringIsConst) && (item->string != NULL))
        {
            global_hooks.deallocate(item->string);
        }
        global_hooks.deallocate(item);
        item = next;
    }
}

/* get the size of the array */
int cJSON_GetArraySize(const cJSON *array)
{
    cJSON *child = NULL;
    size_t size = 0;

    if (array == NULL)
    {
        return 0;
    }

    child = array->child;

    while (child)
    {
        size++;
        child = child->next;
    }

    return (int)size;
}

/* Get item 'index' from the Array 'array'. */
cJSON *cJSON_GetArrayItem(const cJSON *array, int index)
{
    cJSON *child = array ? array->child : NULL;
    while ((child != NULL) && (index > 0))
    {
        index--;
        child = child->next;
    }

    return child;
}

/* Get item 'string' from object. */
cJSON *cJSON_GetObjectItem(const cJSON *const object, const char *const name)
{
    cJSON *child = object ? object->child : NULL;
    while ((child != NULL) && cJSON_strcasecmp((const unsigned char*)child->string, (const unsigned char*)name))
    {
        child = child->next;
    }

    return child;
}

/* Utility for array list handling. */
static void cJSON_AddElementToArray(cJSON *array, cJSON *item)
{
    if (array == NULL || item == NULL)
    {
        return;
    }

    if (array->child == NULL)
    {
        array->child = item;
    }
    else
    {
        cJSON *child = array->child;
        while (child->next)
        {
            child = child->next;
        }
        child->next = item;
        item->prev = child;
    }
}

static void cJSON_AddElementToObject(cJSON *object, const char *name, cJSON *item)
{
    if (object == NULL || name == NULL || item == NULL)
    {
        return;
    }

    item->string = (char*)cJSON_strdup((const unsigned char*)name);
    cJSON_AddElementToArray(object, item);
}

/* Add item to array/object. */
void cJSON_AddItemToArray(cJSON *array, cJSON *item)
{
    cJSON_AddElementToArray(array, item);
}

void cJSON_AddItemToObject(cJSON *object, const char *string, cJSON *item)
{
    cJSON_AddElementToObject(object, string, item);
}

/* Use this when string is definitely const (i.e. a literal, or it will not be freed. */
void cJSON_AddItemToObjectCS(cJSON *object, const char *string, cJSON *item)
{
    if (object == NULL || string == NULL || item == NULL)
    {
        return;
    }

    item->string = (char*)string;
    item->type |= cJSON_StringIsConst;
    cJSON_AddElementToArray(object, item);
}

/* Remove/Detatch items from Arrays/Objects. */
cJSON *cJSON_DetachItemViaPointer(cJSON *parent, cJSON *const item)
{
    if (parent == NULL || item == NULL)
    {
        return NULL;
    }

    if (item->prev)
    {
        item->prev->next = item->next;
    }
    if (item->next)
    {
        item->next->prev = item->prev;
    }
    if (item == parent->child)
    {
        parent->child = item->next;
    }
    item->prev = NULL;
    item->next = NULL;

    return item;
}

cJSON *cJSON_DetachItemFromArray(cJSON *array, int which)
{
    if (array == NULL)
    {
        return NULL;
    }

    return cJSON_DetachItemViaPointer(array, cJSON_GetArrayItem(array, which));
}

void cJSON_DeleteItemFromArray(cJSON *array, int which)
{
    cJSON_Delete(cJSON_DetachItemFromArray(array, which));
}

cJSON *cJSON_DetachItemFromObject(cJSON *object, const char *string)
{
    if (object == NULL || string == NULL)
    {
        return NULL;
    }

    return cJSON_DetachItemViaPointer(object, cJSON_GetObjectItem(object, string));
}

void cJSON_DeleteItemFromObject(cJSON *object, const char *string)
{
    cJSON_Delete(cJSON_DetachItemFromObject(object, string));
}

/* Update array items. */
cJSON_bool cJSON_InsertItemInArray(cJSON *array, int which, cJSON *newitem)
{
    cJSON *child = NULL;
    if (which > cJSON_GetArraySize(array))
    {
        return internal_IsFalse;
    }

    child = cJSON_GetArrayItem(array, which);
    if (child == NULL)
    {
        cJSON_AddItemToArray(array, newitem);
        return internal_IsTrue;
    }

    newitem->next = child;
    newitem->prev = child->prev;
    child->prev = newitem;
    if (child == array->child)
    {
        array->child = newitem;
    }
    else
    {
        newitem->prev->next = newitem;
    }

    return internal_IsTrue;
}

cJSON_bool cJSON_ReplaceItemViaPointer(cJSON *parent, cJSON *const item, cJSON *replacement)
{
    if (parent == NULL || item == NULL || replacement == NULL)
    {
        return internal_IsFalse;
    }

    replacement->next = item->next;
    replacement->prev = item->prev;
    if (replacement->next)
    {
        replacement->next->prev = replacement;
    }
    if (replacement->prev)
    {
        replacement->prev->next = replacement;
    }
    if (parent->child == item)
    {
        parent->child = replacement;
    }

    item->next = NULL;
    item->prev = NULL;
    cJSON_Delete(item);

    return internal_IsTrue;
}

cJSON_bool cJSON_ReplaceItemInArray(cJSON *array, int which, cJSON *newitem)
{
    if (array == NULL || newitem == NULL)
    {
        return internal_IsFalse;
    }
    return cJSON_ReplaceItemViaPointer(array, cJSON_GetArrayItem(array, which), newitem);
}

cJSON_bool cJSON_ReplaceItemInObject(cJSON *object, const char *string, cJSON *newitem)
{
    if (object == NULL || string == NULL || newitem == NULL)
    {
        return internal_IsFalse;
    }
    return cJSON_ReplaceItemViaPointer(object, cJSON_GetObjectItem(object, string), newitem);
}

/* Create basic types: */
cJSON *cJSON_CreateNull(void)
{
    cJSON *item = cJSON_New_Item(&global_hooks);
    if(item)
    {
        item->type = cJSON_NULL;
    }
    return item;
}

cJSON *cJSON_CreateTrue(void)
{
    cJSON *item = cJSON_New_Item(&global_hooks);
    if(item)
    {
        item->type = cJSON_True;
    }
    return item;
}

cJSON *cJSON_CreateFalse(void)
{
    cJSON *item = cJSON_New_Item(&global_hooks);
    if(item)
    {
        item->type = cJSON_False;
    }
    return item;
}

cJSON *cJSON_CreateBool(cJSON_bool b)
{
    cJSON *item = cJSON_New_Item(&global_hooks);
    if(item)
    {
        item->type = b ? cJSON_True : cJSON_False;
    }
    return item;
}

cJSON *cJSON_CreateNumber(double num)
{
    cJSON *item = cJSON_New_Item(&global_hooks);
    if(item)
    {
        item->type = cJSON_Number;
        item->valuedouble = num;
        item->valueint = (int)num;
    }
    return item;
}

cJSON *cJSON_CreateString(const char *string)
{
    cJSON *item = cJSON_New_Item(&global_hooks);
    if(item)
    {
        item->type = cJSON_String;
        item->valuestring = (char*)cJSON_strdup((const unsigned char*)string);
        if(!item->valuestring)
        {
            cJSON_Delete(item);
            return NULL;
        }
    }
    return item;
}

cJSON *cJSON_CreateStringReference(const char *string)
{
    cJSON *item = cJSON_New_Item(&global_hooks);
    if(item)
    {
        item->type = cJSON_String | cJSON_IsReference;
        item->valuestring = (char*)string;
    }
    return item;
}

cJSON *cJSON_CreateArray(void)
{
    cJSON *item = cJSON_New_Item(&global_hooks);
    if(item)
    {
        item->type = cJSON_Array;
    }
    return item;
}

cJSON *cJSON_CreateObject(void)
{
    cJSON *item = cJSON_New_Item(&global_hooks);
    if(item)
    {
        item->type = cJSON_Object;
    }
    return item;
}

/* Create Arrays: */
cJSON *cJSON_CreateIntArray(const int *numbers, int count)
{
    cJSON *n = NULL, *p = NULL, *a = cJSON_CreateArray();
    int i = 0;
    if(!a)
    {
        return NULL;
    }
    for (i = 0; i < count; i++)
    {
        n = cJSON_CreateNumber(numbers[i]);
        if (!n)
        {
            cJSON_Delete(a);
            return NULL;
        }
        if (!i)
        {
            a->child = n;
        }
        else
        {
            p->next = n;
            n->prev = p;
        }
        p = n;
    }
    return a;
}

cJSON *cJSON_CreateFloatArray(const float *numbers, int count)
{
    cJSON *n = NULL, *p = NULL, *a = cJSON_CreateArray();
    int i = 0;
    if(!a)
    {
        return NULL;
    }
    for (i = 0; i < count; i++)
    {
        n = cJSON_CreateNumber(numbers[i]);
        if (!n)
        {
            cJSON_Delete(a);
            return NULL;
        }
        if (!i)
        {
            a->child = n;
        }
        else
        {
            p->next = n;
            n->prev = p;
        }
        p = n;
    }
    return a;
}

cJSON *cJSON_CreateDoubleArray(const double *numbers, int count)
{
    cJSON *n = NULL, *p = NULL, *a = cJSON_CreateArray();
    int i = 0;
    if(!a)
    {
        return NULL;
    }
    for (i = 0; i < count; i++)
    {
        n = cJSON_CreateNumber(numbers[i]);
        if (!n)
        {
            cJSON_Delete(a);
            return NULL;
        }
        if (!i)
        {
            a->child = n;
        }
        else
        {
            p->next = n;
            n->prev = p;
        }
        p = n;
    }
    return a;
}

cJSON *cJSON_CreateStringArray(const char *const *strings, int count)
{
    cJSON *n = NULL, *p = NULL, *a = cJSON_CreateArray();
    int i = 0;
    if(!a)
    {
        return NULL;
    }
    for (i = 0; i < count; i++)
    {
        n = cJSON_CreateString(strings[i]);
        if (!n)
        {
            cJSON_Delete(a);
            return NULL;
        }
        if (!i)
        {
            a->child = n;
        }
        else
        {
            p->next = n;
            n->prev = p;
        }
        p = n;
    }
    return a;
}

/* Duplication */
cJSON *cJSON_Duplicate(const cJSON *item, cJSON_bool recurse)
{
    cJSON *newitem = cJSON_New_Item(&global_hooks);
    if (!newitem)
    {
        return NULL;
    }
    newitem->type = item->type;
    if (item->string)
    {
        newitem->string = (char*)cJSON_strdup((const unsigned char*)item->string);
    }
    if (item->valuestring)
    {
        newitem->valuestring = (char*)cJSON_strdup((const unsigned char*)item->valuestring);
    }
    if (item->child && recurse)
    {
        newitem->child = cJSON_Duplicate(item->child, internal_IsTrue);
        if (!newitem->child)
        {
            cJSON_Delete(newitem);
            return NULL;
        }
    }
    if (item->next && recurse)
    {
        newitem->next = cJSON_Duplicate(item->next, internal_IsTrue);
        if (!newitem->next)
        {
            cJSON_Delete(newitem);
            return NULL;
        }
        newitem->next->prev = newitem;
    }
    return newitem;
}

/* Recursively compare two cJSON objects. */
cJSON_bool cJSON_Compare(const cJSON *const a, const cJSON *const b, cJSON_bool strict)
{
    if ((a == NULL) || (b == NULL) || (a->type != b->type))
    {
        return internal_IsFalse;
    }

    switch (a->type)
    {
        case cJSON_Number:
            if (fabs(a->valuedouble - b->valuedouble) > DBL_EPSILON)
            {
                return internal_IsFalse;
            }
            break;
        case cJSON_String:
            if (cJSON_strcasecmp((const unsigned char*)a->valuestring, (const unsigned char*)b->valuestring))
            {
                return internal_IsFalse;
            }
            break;
        case cJSON_Array:
        case cJSON_Object:
            if (cJSON_GetArraySize(a) != cJSON_GetArraySize(b))
            {
                return internal_IsFalse;
            }
            cJSON *ca = a->child;
            cJSON *cb = b->child;
            while ((ca != NULL) && (cb != NULL))
            {
                if (!cJSON_Compare(ca, cb, strict))
                {
                    return internal_IsFalse;
                }
                ca = ca->next;
                cb = cb->next;
            }
            break;
        default:
            break;
    }

    return internal_IsTrue;
}

/* Parse a cJSON entity from some text and render it to a null-terminated string. */
char *cJSON_ParseAndPrint(const char *input_string)
{
    cJSON *item = cJSON_Parse(input_string);
    if (!item)
    {
        return NULL;
    }
    char *rendered = cJSON_Print(item);
    cJSON_Delete(item);
    return rendered;
}

/* Returns true if the item is a cJSON_Number */
cJSON_bool cJSON_IsNumber(const cJSON *const item)
{
    return item ? (item->type & cJSON_Number) : internal_IsFalse;
}

/* Returns true if the item is a cJSON_String */
cJSON_bool cJSON_IsString(const cJSON *const item)
{
    return item ? (item->type & cJSON_String) : internal_IsFalse;
}

/* Returns true if the item is a cJSON_Array */
cJSON_bool cJSON_IsArray(const cJSON *const item)
{
    return item ? (item->type & cJSON_Array) : internal_IsFalse;
}

/* Returns true if the item is a cJSON_Object */
cJSON_bool cJSON_IsObject(const cJSON *const item)
{
    return item ? (item->type & cJSON_Object) : internal_IsFalse;
}

/* Returns true if the item is a cJSON_True  */
cJSON_bool cJSON_IsBool(const cJSON *const item)
{
    return item ? (item->type & (cJSON_True | cJSON_False)) : internal_IsFalse;
}

/* Returns true if the item is a cJSON_False */
cJSON_bool cJSON_IsFalse(const cJSON *const item)
{
    return item ? (item->type == cJSON_False) : internal_IsFalse;
}

/* Returns true if the item is a cJSON_NULL  */
cJSON_bool cJSON_IsNull(const cJSON *const item)
{
    return item ? (item->type == cJSON_NULL) : internal_IsFalse;
}

/* Returns true if the item is a cJSON_True, cJSON_False or cJSON_NULL */
cJSON_bool cJSON_IsInvalid(const cJSON *const item)
{
    return item ? (item->type == cJSON_Invalid) : internal_IsFalse;
}

/* Returns true if the item is a cJSON_Array, cJSON_Object or cJSON_String */
cJSON_bool cJSON_IsReference_func(const cJSON *const item)
{
    return item ? (item->type & cJSON_IsReference) : internal_IsFalse;
}

/* Returns true if the item is a cJSON_Raw */
cJSON_bool cJSON_IsTrue(const cJSON *const item)
{
    return item ? (item->type == cJSON_True) : internal_IsFalse;
}

/* Get item value from a cJSON_Number object. */
double cJSON_GetNumberValue(const cJSON *const item)
{
    if (!item)
    {
        return 0;
    }
    if (item->type != cJSON_Number)
    {
        return 0;
    }
    return item->valuedouble;
}

/* Get item string from a cJSON_String object. */
const char *cJSON_GetStringValue(const cJSON *const item)
{
    if (!item)
    {
        return NULL;
    }
    if (item->type != cJSON_String)
    {
        return NULL;
    }
    return item->valuestring;
}

#if defined(_MSC_VER)
#pragma warning (pop)
#endif

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif