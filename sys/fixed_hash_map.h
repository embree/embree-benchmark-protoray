// ======================================================================== //
// Copyright 2015-2017 Intel Corporation                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#include <cstring>
#include "common.h"
#include "memory.h"

namespace prt {

inline uint32_t hash(int key)
{
	return key;
}

template <class Key, class Value>
class FixedHashMap : Uncopyable
{
private:
    struct Node
    {
        Key key;
        Value value;
        uint32_t next;
    };

    uint32_t* buckets;
    Node* nodes;
    uint32_t bucketCount;
    uint32_t nodeCount;
    uint32_t maxNodeCount;

public:
    FixedHashMap() : buckets(0), nodes(0), bucketCount(0), nodeCount(1) {}

	FixedHashMap(int capacity, int bucketCount)
	{
		assert(capacity >= 0);
		assert(bucketCount >= 0);

        init(capacity, bucketCount);
	}

	~FixedHashMap()
	{
        alignedFree(buckets);
        alignedFree(nodes);
	}

    void alloc(int capacity, int bucketCount)
	{
		assert(capacity >= 0);
		assert(bucketCount >= 0);

        alignedFree(buckets);
        alignedFree(nodes);

        init(capacity, bucketCount);
	}

	void free()
	{
        alignedFree(buckets);
        alignedFree(nodes);

        buckets = 0;
        nodes = 0;

        nodeCount = 1;
	}

    Value& operator [](const Key& key)
	{
        uint32_t bucketId = (uint32_t)hash(key) % bucketCount;

		// Is the bucket empty?
        if (buckets[bucketId] == 0)
		{
            buckets[bucketId] = nodeCount;

            new (&nodes[nodeCount].key) Key(key);
            new (&nodes[nodeCount].value) Value;
            nodes[nodeCount].next = 0;
			
            return nodes[nodeCount++].value;
		}

		// Look for the key in the linked list
        uint32_t nodeId = buckets[bucketId];

		for (; ;)
		{
            if (nodes[nodeId].key == key)
                return nodes[nodeId].value;

			// Is this the last node of the list?
            if (nodes[nodeId].next == 0)
				break;

            nodeId = nodes[nodeId].next;
		}

		// Append a new node to the list
        if (nodeCount == maxNodeCount)
            throw std::runtime_error("FixedHashMap: not enough memory to store entry");

        nodes[nodeId].next = nodeCount;

        new (&nodes[nodeCount].key) Key(key);
        new (&nodes[nodeCount].value) Value;
        nodes[nodeCount].next = 0;
		
        return nodes[nodeCount++].value;
	}

	// Returns true if the pair was added
	// If the pair could not be added, the existingValue is set
    bool add(const Key& key, const Value& value, Value& existingValue)
	{
        uint32_t bucketId = (uint32_t)hash(key) % bucketCount;

		// Is the bucket empty?
        if (buckets[bucketId] == 0)
		{
            if (nodeCount == maxNodeCount)
                throw std::runtime_error("FixedHashMap: not enough memory to store entry");

            buckets[bucketId] = nodeCount;

            new (&nodes[nodeCount].key) Key(key);
            new (&nodes[nodeCount].value) Value(value);
            nodes[nodeCount].next = 0;
			
            ++nodeCount;
			return true;
		}

		// Look for the key in the linked list
        uint32_t nodeId = buckets[bucketId];

		for (; ;)
		{
            if (nodes[nodeId].key == key)
			{
                existingValue = nodes[nodeId].value;
				return false;
			}

			// Is this the last node of the list?
            if (nodes[nodeId].next == 0)
				break;

            nodeId = nodes[nodeId].next;
		}

		// Append a new node to the list
        if (nodeCount == maxNodeCount)
            throw std::runtime_error("FixedHashMap: not enough memory to store entry");

        nodes[nodeId].next = nodeCount;

        new (&nodes[nodeCount].key) Key(key);
        new (&nodes[nodeCount].value) Value(value);
        nodes[nodeCount].next = 0;
		
        ++nodeCount;
		return true;
	}

	void clear()
	{
        memset(buckets, 0, bucketCount * sizeof(uint32_t));

        for (uint32_t i = 0; i < nodeCount; ++i)
		{
            nodes[i].key.~Key();
            nodes[i].value.~Value();
		}

		// The first node is a dummy node, because 0 is the invalid node
        nodeCount = 1;
	}

    int getSize() const
	{
        return nodeCount - 1;
	}

private:
    void init(int capacity, int bucketCount)
	{
        this->bucketCount = bucketCount;
        buckets = (uint32_t*)alignedAlloc(bucketCount * sizeof(uint32_t), cacheLineSize);
        maxNodeCount = capacity + 1;
        nodes = (Node*)alignedAlloc(maxNodeCount * sizeof(Node), cacheLineSize);

		clear();
	}
};

} // namespace prt
