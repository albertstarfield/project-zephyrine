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

#ifndef cJSON__h
#define cJSON__h

#ifdef __cplusplus
extern "C"
{
#endif

#if !defined(__WINDOWS__) && (defined(WIN32) || defined(WIN64) || defined(_MSC_VER) || defined(_WIN32))
#define __WINDOWS__
#endif

#ifdef __WINDOWS__
/* Disable warnings about unused parameters on Windows. */
#pragma warning (push)
#pragma warning (disable : 4100)
#endif

/* project version */
#define CJSON_VERSION_MAJOR 1
#define CJSON_VERSION_MINOR 7
#define CJSON_VERSION_PATCH 16

#include <stddef.h>

/* cJSON Types: */
#define cJSON_Invalid (0)
#define cJSON_False  (1 << 0)
#define cJSON_True   (1 << 1)
#define cJSON_NULL   (1 << 2)
#define cJSON_Number (1 << 3)
#define cJSON_String (1 << 4)
#define cJSON_Array  (1 << 5)
#define cJSON_Object (1 << 6)

#define cJSON_Raw    (1 << 7) /* raw json */

#define cJSON_IsReference 256
#define cJSON_StringIsConst 512

/* The cJSON structure: */
typedef struct cJSON
{
    /* next/prev allow you to walk array/object chains. Alternatively, use GetArraySize/GetArrayItem/GetObjectItem */
    struct cJSON *next;
    struct cJSON *prev;
    /* An array or object item will have its name here. */
    char *string;
    int type;
    /* Our values: */
    union {
        double valuedouble;
        int valueint;
    };
    /* For other items, this string holds their value. */
    char *valuestring;

    /* child indicates how many items in the object/array we have. */
    struct cJSON *child;
} cJSON;

typedef struct cJSON_Hooks
{
    void *(*malloc_fn)(size_t sz);
    void (*free_fn)(void *ptr);
} cJSON_Hooks;

typedef int cJSON_bool;

/* Supply a block of JSON, and this returns a cJSON object you can interrogate. Call cJSON_Delete when finished. */
cJSON *cJSON_Parse(const char *value);
/* ParseWithOpts allows specifying options - see cJSON_ParseWithOpts for details. */
cJSON *cJSON_ParseWithOpts(const char *value, const char **return_parse_end, cJSON_bool require_null_terminated);

/* Render a cJSON object in text for transfer/storage. Free the resultant string with cJSON_Delete. */
char *cJSON_Print(const cJSON *item);
/* Render a cJSON object to text for transfer/storage without any formatting. Free the resultant string with cJSON_Delete. */
char *cJSON_PrintUnformatted(const cJSON *item);
/* Render a cJSON object to text using a buffered strategy. prebuffer is a initial buffer size.
 * Postbuffer is the growth factor. Free the resultant string with cJSON_Delete. */
char *cJSON_PrintBuffered(const cJSON *item, int prebuffer, cJSON_bool fmt);
/* Render a cJSON object to file (fprintf)
 * If close_file is true, the file is closed. */
cJSON_bool cJSON_PrintToFile(cJSON *item, const char *filename, cJSON_bool close_file);

/* Delete a cJSON entity and all its children. */
void cJSON_Delete(cJSON *item);

/* Returns the number of items in an array (or object). */
int cJSON_GetArraySize(const cJSON *array);
/* Retrieve item number "index" from objects/arrays. */
cJSON *cJSON_GetArrayItem(const cJSON *array, int index);
/* Get item "string" from object. Case insensitive. */
cJSON *cJSON_GetObjectItem(const cJSON *const object, const char *const string);
cJSON *cJSON_GetObjectItemCaseSensitive(const cJSON *const object, const char *const string);
cJSON_bool cJSON_HasObjectItem(const cJSON *object, const char *string);
/* Utility for array list handling. */
void cJSON_AddItemToArray(cJSON *array, cJSON *item);
void cJSON_AddItemToObject(cJSON *object, const char *string, cJSON *item);
/* Use this when string is definitely const (i.e. a literal, or it will not be freed. */
void cJSON_AddItemToObjectCS(cJSON *object, const char *string, cJSON *item);
/* Remove/Detatch items from Arrays/Objects. */
cJSON *cJSON_DetachItemViaPointer(cJSON *parent, cJSON *const item);
cJSON *cJSON_DetachItemFromArray(cJSON *array, int which);
void cJSON_DeleteItemFromArray(cJSON *array, int which);
cJSON *cJSON_DetachItemFromObject(cJSON *object, const char *string);
cJSON *cJSON_DetachItemFromObjectCaseSensitive(cJSON *object, const char *string);
void cJSON_DeleteItemFromObject(cJSON *object, const char *string);
void cJSON_DeleteItemFromObjectCaseSensitive(cJSON *object, const char *string);

/* Update array items. */
cJSON_bool cJSON_InsertItemInArray(cJSON *array, int which, cJSON *newitem);
cJSON_bool cJSON_ReplaceItemViaPointer(cJSON *parent, cJSON *const item, cJSON *replacement);
cJSON_bool cJSON_ReplaceItemInArray(cJSON *array, int which, cJSON *newitem);
cJSON_bool cJSON_ReplaceItemInObject(cJSON *object, const char *string, cJSON *newitem);
cJSON_bool cJSON_ReplaceItemInObjectCaseSensitive(cJSON *object, const char *string, cJSON *newitem);

/* Create types: */
cJSON *cJSON_CreateNull(void);
cJSON *cJSON_CreateTrue(void);
cJSON *cJSON_CreateFalse(void);
cJSON *cJSON_CreateBool(cJSON_bool boolean);
cJSON *cJSON_CreateNumber(double num);
cJSON *cJSON_CreateString(const char *string);
/* Create a string where the string is not created with malloc. */
cJSON *cJSON_CreateStringReference(const char *string);
cJSON *cJSON_CreateArray(void);
cJSON *cJSON_CreateObject(void);

/* Create Arrays: */
cJSON *cJSON_CreateIntArray(const int *numbers, int count);
cJSON *cJSON_CreateFloatArray(const float *numbers, int count);
cJSON *cJSON_CreateDoubleArray(const double *numbers, int count);
cJSON *cJSON_CreateStringArray(const char *const *strings, int count);

/* Duplication */
cJSON *cJSON_Duplicate(const cJSON *item, cJSON_bool recurse);
/* Recursively compare two cJSON objects. */
cJSON_bool cJSON_Compare(const cJSON *const a, const cJSON *const b, cJSON_bool strict);

/* Parse a cJSON entity from some text and render it to a null-terminated string. */
char *cJSON_ParseAndPrint(const char *input_string);

/* Supply a user-defined set of callbacks for malloc/free. */
void cJSON_InitHooks(cJSON_Hooks* hooks);

/* Reset the cJSON_Hooks structure to the default malloc/free functions. */
void cJSON_ResetHooks(void);

/* Returns true if the item is a cJSON_Number */
cJSON_bool cJSON_IsNumber(const cJSON *const item);
/* Returns true if the item is a cJSON_String */
cJSON_bool cJSON_IsString(const cJSON *const item);
/* Returns true if the item is a cJSON_Array */
cJSON_bool cJSON_IsArray(const cJSON *const item);
/* Returns true if the item is a cJSON_Object */
cJSON_bool cJSON_IsObject(const cJSON *const item);
/* Returns true if the item is a cJSON_True  */
cJSON_bool cJSON_IsBool(const cJSON *const item);
/* Returns true if the item is a cJSON_False */
cJSON_bool cJSON_IsFalse(const cJSON *const item);
/* Returns true if the item is a cJSON_NULL  */
cJSON_bool cJSON_IsNull(const cJSON *const item);
/* Returns true if the item is a cJSON_True, cJSON_False or cJSON_NULL */
cJSON_bool cJSON_IsInvalid(const cJSON *const item);
/* Returns true if the item is a cJSON_Array, cJSON_Object or cJSON_String */
cJSON_bool cJSON_IsReference_func(const cJSON *const item);
/* Returns true if the item is a cJSON_Raw */
cJSON_bool cJSON_IsTrue(const cJSON *const item);

/* Get item value from a cJSON_Number object. */
double cJSON_GetNumberValue(const cJSON *const item);
/* Get item string from a cJSON_String object. */
const char *cJSON_GetStringValue(const cJSON *const item);

/* These utilities are only available if cJSON_UnitTests is defined */
#ifdef CJSON_TEST
/* Read a file into a string and return it. Free with cJSON_free. */
char* cJSON_ReadFile(const char* filename);
/* Write a string to a file. */
cJSON_bool cJSON_WriteFile(const char* filename, const char* string);
#endif

/* When compiling for windows, we can disable deprecation warnings for strcpy, sprintf, etc. */
#ifdef __WINDOWS__
#pragma warning (pop)
#endif

#ifdef __cplusplus
}
#endif

#endif
