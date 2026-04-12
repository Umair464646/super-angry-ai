import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Rectangle {
    id: root
    property var entries: []
    property bool expanded: true

    color: "#0E1420"
    border.color: "#1A2638"
    radius: 12

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 8
        spacing: 8

        RowLayout {
            Layout.fillWidth: true
            Label { text: "Logs"; color: "#D8E7FF"; font.bold: true }
            Item { Layout.fillWidth: true }
            Button {
                text: root.expanded ? "Collapse" : "Expand"
                onClicked: root.expanded = !root.expanded
            }
        }

        ListView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true
            model: root.expanded ? root.entries : []
            delegate: Label {
                required property var modelData
                text: "[" + modelData.ts + "][" + modelData.level + "] " + modelData.msg
                color: modelData.level === "WARN" ? "#F6C77C" : "#A8C6E8"
                font.pixelSize: 12
                wrapMode: Text.Wrap
            }
            ScrollBar.vertical: ScrollBar {}
        }
    }
}
