package com.battlesbudz.phoneinference

/**
 * Template captioning over the four model outputs. There is no language
 * model in this repo; this builds a 2-4 sentence plain-language summary
 * from counts and labels (the same fusion style as the Jarvis bus-photo
 * test), not generative text.
 */
object CaptionBuilder {

    private val VEHICLE_PLURALS = mapOf(
        "car" to "cars",
        "bus" to "buses",
        "truck" to "trucks",
        "motorcycle" to "motorcycles",
        "bicycle" to "bicycles",
        "airplane" to "airplanes",
        "train" to "trains",
        "boat" to "boats",
    )

    /** A few ImageNet labels that read as a setting; anything else is omitted. */
    private val SCENE_HINTS = mapOf(
        "streetcar" to "a street with a tram",
        "seashore" to "a beach or shoreline",
        "lakeside" to "a lakeside",
        "restaurant" to "a restaurant",
        "grocery store" to "a grocery store",
        "church" to "a church",
        "parking meter" to "a street with parking meters",
    )

    fun build(
        detections: List<Detection>,
        segments: List<SegResult>,
        poses: List<PoseResult>,
        classes: List<ClassPrediction>,
    ): String {
        val sentences = mutableListOf<String>()

        // People: pose is the stronger signal; detection "person" catches
        // people the pose model misses (occlusion, tiny figures).
        val posePeople = poses.size
        val detPeople = detections.count { it.label == "person" }
        val people = maxOf(posePeople, detPeople)

        val vehicleParts = VEHICLE_PLURALS.mapNotNull { (singular, plural) ->
            val n = detections.count { it.label == singular }
            if (n > 0) qty(n, singular, plural) else null
        }

        val parts = mutableListOf<String>()
        if (people == 1) {
            parts.add("1 person")
        } else if (people > 1) {
            parts.add("about ${qty(people, "person", "people")}")
        }
        parts.addAll(vehicleParts)
        sentences.add(
            if (parts.isEmpty()) "I don't see any people or vehicles clearly."
            else "I see ${joinAnd(parts)}."
        )

        // Other notable objects at >= 0.5 confidence, excluding what we counted.
        val counted = setOf("person") + VEHICLE_PLURALS.keys
        val notable = detections
            .filter { it.label !in counted && it.confidence >= 0.5f }
            .distinctBy { it.label }
            .take(4)
            .map { "a ${it.label}" }
        if (notable.isNotEmpty()) {
            sentences.add("Also visible: ${joinAnd(notable)}.")
        }

        // Scene hint from the classifier, only when it maps to a setting.
        val top = classes.firstOrNull()
        val scene = top?.let { SCENE_HINTS[it.label] }
        if (scene != null && (top?.confidence ?: 0f) >= 0.25f) {
            sentences.add("The setting looks like $scene.")
        }

        return sentences.joinToString(" ")
    }

    private fun qty(n: Int, singular: String, plural: String): String =
        "$n ${if (n == 1) singular else plural}"

    private fun joinAnd(items: List<String>): String = when (items.size) {
        0 -> ""
        1 -> items[0]
        2 -> "${items[0]} and ${items[1]}"
        else -> items.dropLast(1).joinToString(", ") + ", and " + items.last()
    }
}
