import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Rectangle {
    id: root
    required property var row
    property bool selected: false
    signal clicked()

    height: 86
    radius: 10
    color: selected ? "#123052" : (hovered ? "#111C2D" : "#0E1623")
    border.color: selected ? "#36A3FF" : "#1B2A41"
    property bool hovered: false

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 10
        spacing: 4
        RowLayout {
            Layout.fillWidth: true
            Label { text: row.id + " · " + row.name; color: "#E3EEFF"; font.bold: true; Layout.fillWidth: true; elide: Text.ElideRight }
            Label { text: row.status; color: "#8FCBFF"; font.pixelSize: 11 }
        }
        Label { text: row.family + " | G" + row.generation + " | " + row.origin; color: "#9FB5D7"; font.pixelSize: 12 }
        RowLayout {
            Label { text: "Fit " + row.fitness; color: "#D3E7FF"; font.pixelSize: 12 }
            Label { text: "Rob " + row.robustness; color: "#8CE0C9"; font.pixelSize: 12 }
            Label { text: "Val " + row.validation; color: "#EACB95"; font.pixelSize: 12 }
        }
    }

    MouseArea {
        anchors.fill: parent
        hoverEnabled: true
        onEntered: root.hovered = true
        onExited: root.hovered = false
        onClicked: root.clicked()
    }
}
